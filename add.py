#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF OCR 付加バッチ（日本語向け / YomiToku版）
- 選択フォルダ内の PDF を処理
- 既にテキスト層（OCR/埋め込み文字）がある PDF はスキップ（無加工）
- テキスト層が無い PDF のみ OCR を実行し、`_ocr.pdf` を出力
- 同名出力がある場合は（上書きOFF時） `_ocr_001.pdf` のように連番保存

前提:
- OCR エンジン: YomiToku CLI
- 日本語文書を主対象
- 読み順は現行用途を踏まえ `top2bottom`（横書き向け）を既定指定
"""

from __future__ import annotations

import csv
import os
import re
import sys
import shutil
import queue
import threading
import subprocess
import tempfile
import time
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Tkinter の読み込みに失敗しました: {e}")

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore

APP_TITLE = "PDF OCR 付加バッチ（日本語・横書き / YomiToku）"
OUTPUT_SUFFIX = "_ocr"
GENERATED_INPUT_NAME_RE = re.compile(rf"{re.escape(OUTPUT_SUFFIX)}(?:_\d{{3,}})?$", re.IGNORECASE)
DEFAULT_YOMITOKU_TIMEOUT_SEC = 60 * 60  # 1時間。ハング対策の安全弁


@dataclass
class ProcessResult:
    total: int = 0
    skipped_has_text: int = 0
    skipped_name_rule: int = 0
    processed: int = 0
    errors: int = 0


@dataclass(frozen=True)
class RunConfig:
    folder: Path
    recursive: bool
    overwrite: bool
    same_folder_output: bool
    write_csv_log: bool
    selected_mode: str  # auto / cpu / gpu (UI値)
    resolved_device: str  # cpu / cuda (実行値)
    lite: bool
    dpi: int
    yomitoku_timeout_sec: int = DEFAULT_YOMITOKU_TIMEOUT_SEC


@dataclass(frozen=True)
class RunSummary:
    started_at: datetime
    ended_at: datetime
    result: ProcessResult
    was_stopped: bool


@dataclass(frozen=True)
class BatchRunOutcome:
    final_status: str
    ui_current: int
    ui_total: int


@dataclass(frozen=True)
class FinishUiState:
    status: str
    current: int
    total: int


@dataclass(frozen=True)
class BatchCompletionPresentation:
    final_log_kind: str
    final_log_message: str
    outcome: BatchRunOutcome


@dataclass(frozen=True)
class StartRunPlan:
    run_config: RunConfig
    mode_label: str
    startup_log_message: str
    pre_logs: tuple[tuple[str, str], ...]
    gpu_warning_message: Optional[str] = None


class StartRunCoordinator:
    """on_start の事前準備（検証 / デバイス解決 / RunConfig生成）をまとめる。"""

    def __init__(
        self,
        *,
        validate_runtime_settings: Callable[[str], tuple[bool, str]],
        check_dependencies: Callable[[bool], tuple[bool, str]],
        resolve_device_for_run: Callable[[str], str],
        check_cuda_available: Callable[[], tuple[bool, str]],
    ) -> None:
        self._validate_runtime_settings = validate_runtime_settings
        self._check_dependencies = check_dependencies
        self._resolve_device_for_run = resolve_device_for_run
        self._check_cuda_available = check_cuda_available

    def prepare(
        self,
        *,
        folder_str: str,
        recursive: bool,
        overwrite: bool,
        same_folder_output: bool,
        write_csv_log: bool,
        selected_mode_raw: str,
        lite: bool,
        dpi_text: str,
    ) -> tuple[bool, str, Optional[StartRunPlan]]:
        folder_str = (folder_str or "").strip()
        if not folder_str:
            return False, "対象フォルダを選択してください。", None

        folder = Path(folder_str)
        if not folder.exists():
            return False, "指定フォルダが存在しません。", None
        if not folder.is_dir():
            return False, "指定パスはフォルダではありません。", None

        ok, msg = self._validate_runtime_settings(dpi_text)
        if not ok:
            return False, msg, None

        deps_ok, dep_msg = self._check_dependencies(True)
        if not deps_ok:
            return False, dep_msg, None

        selected_mode = (selected_mode_raw or "auto").lower()
        resolved_device = self._resolve_device_for_run(selected_mode)

        pre_logs: list[tuple[str, str]] = []
        gpu_warning_message: Optional[str] = None
        if selected_mode in ("auto", "gpu"):
            cuda_ok, cuda_msg = self._check_cuda_available()
            if selected_mode == "gpu" and not cuda_ok:
                gpu_warning_message = (
                    "GPU(CUDA)が利用できない可能性があります。\n\n"
                    + cuda_msg
                    + "\n\nこのまま実行すると失敗、またはYomiToku側でCPU動作になる場合があります。"
                )
                pre_logs.append(("WARN", f"GPU事前確認: {cuda_msg}"))
            elif selected_mode == "auto":
                if cuda_ok:
                    pre_logs.append(("INFO", "自動モード: GPU(CUDA) を使用します。"))
                else:
                    pre_logs.append(("INFO", f"自動モード: CPU を使用します（GPU未利用: {cuda_msg}）"))

        try:
            dpi = int(dpi_text)
        except Exception:
            return False, "PDF読取DPIは整数で指定してください。", None

        run_config = RunConfig(
            folder=folder,
            recursive=bool(recursive),
            overwrite=bool(overwrite),
            same_folder_output=bool(same_folder_output),
            write_csv_log=bool(write_csv_log),
            selected_mode=selected_mode,
            resolved_device=resolved_device,
            lite=bool(lite),
            dpi=dpi,
        )

        mode_label = {"auto": "自動", "cpu": "CPU", "gpu": "GPU(CUDA)"}.get(selected_mode, selected_mode)
        startup_log_message = (
            f"処理を開始します。（エンジン: YomiToku / 実行デバイス: {mode_label} -> {resolved_device}）"
        )

        plan = StartRunPlan(
            run_config=run_config,
            mode_label=mode_label,
            startup_log_message=startup_log_message,
            pre_logs=tuple(pre_logs),
            gpu_warning_message=gpu_warning_message,
        )
        return True, "", plan


class CsvLogger:
    HEADERS = ["timestamp", "input", "output", "action", "detail", "device_selected", "device_resolved", "seconds"]

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._rows.clear()

    def new_row(self, pdf_path: Path, run_config: RunConfig) -> dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input": str(pdf_path),
            "output": "",
            "action": "",
            "detail": "",
            "device_selected": run_config.selected_mode,
            "device_resolved": run_config.resolved_device,
            "seconds": "",
        }

    def append_row(self, row: dict[str, Any]) -> None:
        self._rows.append(dict(row))

    def write(self, csv_path: Path, run_config: RunConfig, summary: RunSummary) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        result = summary.result
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            # メタ情報（冒頭に記録）
            w.writerow(["meta", "app_title", APP_TITLE])
            w.writerow(["meta", "target_folder", str(run_config.folder)])
            w.writerow(["meta", "started_at", summary.started_at.isoformat(timespec="seconds")])
            w.writerow(["meta", "ended_at", summary.ended_at.isoformat(timespec="seconds")])
            w.writerow(["meta", "device_selected", run_config.selected_mode])
            w.writerow(["meta", "device_resolved", run_config.resolved_device])
            w.writerow(["meta", "recursive", str(run_config.recursive)])
            w.writerow(["meta", "overwrite", str(run_config.overwrite)])
            w.writerow(["meta", "same_folder_output", str(run_config.same_folder_output)])
            w.writerow(["meta", "lite", str(run_config.lite)])
            w.writerow(["meta", "dpi", str(run_config.dpi)])
            w.writerow(["meta", "stopped", str(bool(summary.was_stopped))])
            w.writerow(["meta", "total", result.total])
            w.writerow(["meta", "processed", result.processed])
            w.writerow(["meta", "skipped_has_text", result.skipped_has_text])
            w.writerow(["meta", "skipped_name_rule", result.skipped_name_rule])
            w.writerow(["meta", "errors", result.errors])
            w.writerow([])

            w.writerow(self.HEADERS)
            for row in self._rows:
                w.writerow([row.get(h, "") for h in self.HEADERS])



class YomiTokuRunner:
    """YomiToku 呼び出し責務をまとめた実行ヘルパー（同一ファイル内の段階分離）。"""

    def __init__(
        self,
        *,
        resolve_launcher: Callable[[], list[str]],
        reset_launcher_cache: Callable[[], None],
        log: Callable[[str, str], None],
        is_stop_requested: Callable[[], bool],
        terminate_process: Callable[[Any], None],
        set_active_process: Callable[[Any], None],
        quote_for_log: Callable[[str], str],
    ) -> None:
        self._resolve_launcher = resolve_launcher
        self._reset_launcher_cache = reset_launcher_cache
        self._log = log
        self._is_stop_requested = is_stop_requested
        self._terminate_process = terminate_process
        self._set_active_process = set_active_process
        self._quote_for_log = quote_for_log

    def build_cmd(
        self,
        in_pdf: Path,
        outdir: Path,
        dpi: int,
        device: str,
        lite: bool,
        launcher: Optional[list[str]] = None,
    ) -> list[str]:
        cmd = (launcher or self._resolve_launcher()) + [
            str(in_pdf),
            "-o", str(outdir),
            "-f", "pdf",
            "--combine",
            "--dpi", str(dpi),
            "-d", device,
            "--reading_order", "top2bottom",
        ]
        if lite:
            cmd.append("--lite")
        return cmd

    def find_generated_pdf(self, outdir: Path, input_pdf: Optional[Path] = None) -> tuple[Optional[Path], str]:
        """YomiToku 出力PDFを候補優先化して選ぶ。

        返り値: (選択PDF or None, 選択理由ログ文字列)
        """
        pdfs = [p for p in outdir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
        if not pdfs:
            return None, "candidate_count=0"
        if len(pdfs) == 1:
            rel = str(pdfs[0].relative_to(outdir))
            return pdfs[0], f"candidate_count=1; selected={rel}; reason=only_candidate"

        input_stem_tokens: set[str] = set()
        if input_pdf is not None:
            input_stem_tokens = {t for t in re.split(r"[^a-z0-9]+", input_pdf.stem.lower()) if t}

        scored: list[tuple[tuple[int, int, int, int, float], Path, dict[str, Any]]] = []
        for p in pdfs:
            rel = str(p.relative_to(outdir))
            stem_lower = p.stem.lower()
            name_tokens = {t for t in re.split(r"[^a-z0-9]+", stem_lower) if t}
            has_combine = 1 if "combine" in stem_lower else 0
            token_overlap = len(input_stem_tokens & name_tokens) if input_stem_tokens else 0
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            try:
                mt = p.stat().st_mtime
            except Exception:
                mt = 0.0
            depth_penalty = -len(p.relative_to(outdir).parts)
            score = (has_combine, token_overlap, depth_penalty, int(size), mt)
            meta = {
                "rel": rel,
                "combine": bool(has_combine),
                "token_overlap": token_overlap,
                "size": int(size),
                "mtime": float(mt),
            }
            scored.append((score, p, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        winner = scored[0]
        top_meta = winner[2]
        preview_parts = []
        for _, _, meta in scored[:5]:
            preview_parts.append(
                f"{meta['rel']}{{combine={int(meta['combine'])},overlap={meta['token_overlap']},size={meta['size']}}}"
            )
        reason = (
            f"candidate_count={len(scored)}; "
            f"selected={top_meta['rel']}; "
            f"rule=combine>name_token_overlap>shallower_path>size>mtime; "
            f"top_candidates=[{' | '.join(preview_parts)}]"
        )
        return winner[1], reason

    def run_subprocess_with_polling(self, cmd: list[str], timeout_sec: int) -> tuple[int, str, str]:
        popen_kwargs: dict[str, Any] = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if creationflags:
                popen_kwargs["creationflags"] = creationflags
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)
        self._set_active_process(proc)
        started = time.monotonic()

        try:
            while True:
                if self._is_stop_requested():
                    self._terminate_process(proc)
                    try:
                        proc.communicate(timeout=2)
                    except Exception:
                        pass
                    raise OCRStopRequested("停止要求によりOCR実行を中断しました")

                rc = proc.poll()
                if rc is not None:
                    stdout, stderr = proc.communicate()
                    return rc, (stdout or ""), (stderr or "")

                if timeout_sec > 0 and (time.monotonic() - started) > timeout_sec:
                    self._terminate_process(proc)
                    try:
                        proc.communicate(timeout=2)
                    except Exception:
                        pass
                    raise RuntimeError(f"YomiToku 実行がタイムアウトしました（>{timeout_sec}秒）")

                time.sleep(0.2)
        finally:
            self._set_active_process(None)

    def run(self, in_pdf: Path, out_pdf: Path, dpi: int, device: str, lite: bool, timeout_sec: int) -> None:
        if out_pdf.exists():
            try:
                out_pdf.unlink()
            except Exception as e:
                raise RuntimeError(f"既存出力ファイル削除失敗: {e}")

        with tempfile.TemporaryDirectory(prefix="yomitoku_out_") as td:
            outdir = Path(td)
            launcher = self._resolve_launcher()
            cmd = self.build_cmd(in_pdf, outdir, dpi=dpi, device=device, lite=lite, launcher=launcher)
            self._log("INFO", "[CMD] " + " ".join(self._quote_for_log(x) for x in cmd))

            rc, stdout, stderr = self.run_subprocess_with_polling(cmd, timeout_sec=timeout_sec)

            if rc != 0:
                detail_all = ((stderr or "") + "\n" + (stdout or "")).strip()
                if "No module named yomitoku.__main__" in detail_all:
                    self._reset_launcher_cache()
                    launcher2 = self._resolve_launcher()
                    if launcher2 != launcher:
                        cmd2 = self.build_cmd(in_pdf, outdir, dpi=dpi, device=device, lite=lite, launcher=launcher2)
                        self._log("WARN", "[RETRY CMD] " + " ".join(self._quote_for_log(x) for x in cmd2))
                        rc, stdout, stderr = self.run_subprocess_with_polling(cmd2, timeout_sec=timeout_sec)

            if rc != 0:
                detail = (stderr or "").strip() or (stdout or "").strip()
                detail = (detail[:1500] + "...") if len(detail) > 1500 else detail
                raise RuntimeError(f"YomiToku 失敗 (code={rc}) {detail}")

            generated, selected_reason = self.find_generated_pdf(outdir, input_pdf=in_pdf)
            if generated is None:
                files = sorted([str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()])
                head = ", ".join(files[:20]) if files else "(なし)"
                if len(files) > 20:
                    head += ", ..."
                raise RuntimeError(
                    "YomiToku 実行は成功しましたが、出力PDFを検出できませんでした。"
                    f" 出力先: {outdir} / 生成物: {head} / 判定: {selected_reason}"
                )
            self._log("INFO", f"[PICK] {selected_reason}")

            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(generated, out_pdf)
            except Exception as e:
                raise RuntimeError(f"出力PDFのコピーに失敗しました: {e}")


class PdfInspector:
    """PDF収集とテキスト層判定の責務をまとめるヘルパー（同一ファイル内の段階分離）。"""

    def __init__(self, *, pdf_reader_cls: Any, log: Callable[[str, str], None]) -> None:
        self._pdf_reader_cls = pdf_reader_cls
        self._log = log

    def collect_pdfs(self, folder: Path, recursive: bool) -> List[Path]:
        if recursive:
            files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
        else:
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        return sorted(set(files), key=lambda p: str(p).lower())

    def has_text_layer(self, pdf_path: Path, pages_to_check: int = 5, min_chars: int = 10) -> bool:
        if self._pdf_reader_cls is None:
            raise RuntimeError("pypdf が利用できません")

        try:
            reader = self._pdf_reader_cls(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"PDF 読み込み失敗: {e}")

        try:
            if getattr(reader, "is_encrypted", False):
                try:
                    dec_result = reader.decrypt("")
                except Exception as e:
                    self._log("WARN", f"[PDF-CHECK] 暗号化PDFの判定失敗: {pdf_path.name} ({e})")
                    raise RuntimeError("暗号化PDFのため判定できません（パスワード解除後に再実行してください）")
                if not dec_result:
                    self._log("WARN", f"[PDF-CHECK] 暗号化PDFの判定不可: {pdf_path.name} (decrypt returned falsey)")
                    raise RuntimeError("暗号化PDFのため判定できません（パスワード解除後に再実行してください）")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"PDFテキスト層判定中にエラー: {e}")

        chars = 0
        check_pages = min(len(reader.pages), pages_to_check)
        for i in range(check_pages):
            try:
                txt = reader.pages[i].extract_text() or ""
            except Exception:
                txt = ""
            txt = re.sub(r"\s+", "", txt)
            chars += len(txt)
            if chars >= min_chars:
                return True
        return False


class OutputPathResolver:
    """出力先パス決定（同名回避含む）の責務をまとめるヘルパー。"""

    def default_output_path(self, pdf_path: Path, same_folder_output: bool) -> Path:
        if same_folder_output:
            return pdf_path.with_name(pdf_path.stem + OUTPUT_SUFFIX + pdf_path.suffix)
        out_dir = pdf_path.parent / "ocr_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / (pdf_path.stem + OUTPUT_SUFFIX + pdf_path.suffix)

    def make_output_path(self, pdf_path: Path, same_folder_output: bool, overwrite: bool) -> Path:
        base = self.default_output_path(pdf_path, same_folder_output)
        if overwrite or (not base.exists()):
            return base

        parent = base.parent
        stem = base.stem  # includes _ocr
        suffix = base.suffix
        for i in range(1, 10000):
            cand = parent / f"{stem}_{i:03d}{suffix}"
            if not cand.exists():
                return cand
        raise RuntimeError("連番出力先を確保できませんでした（_001〜_9999 が埋まっています）")


class RuntimeSupport:
    """実行時バリデーション/依存関係/CUDA/ランチャー解決をまとめるヘルパー。"""

    def __init__(self, *, has_pypdf: bool) -> None:
        self._has_pypdf = bool(has_pypdf)
        self._yomitoku_launcher_cache: Optional[list[str]] = None

    def reset_yomitoku_launcher_cache(self) -> None:
        self._yomitoku_launcher_cache = None

    def validate_runtime_settings(self, dpi_text: str) -> tuple[bool, str]:
        try:
            dpi = int(str(dpi_text).strip())
            if dpi < 72 or dpi > 1200:
                return False, "PDF読取DPI は 72〜1200 の範囲で指定してください。"
        except ValueError:
            return False, "PDF読取DPI は整数で入力してください。"
        return True, ""

    def check_dependencies(self, return_message: bool = False) -> tuple[bool, str]:
        problems: List[str] = []

        if not self._has_pypdf:
            problems.append("- Python ライブラリ `pypdf` が見つかりません（`py -m pip install pypdf`）")

        try:
            import importlib.util
            has_module = importlib.util.find_spec("yomitoku") is not None
        except Exception:
            has_module = False

        has_cli = shutil.which("yomitoku") is not None
        scripts_hit = self.find_local_yomitoku_script() is not None
        if not (has_module or has_cli or scripts_hit):
            problems.append("- `yomitoku` が見つかりません（`py -m pip install yomitoku`）")

        if not problems:
            try:
                _ = self.resolve_yomitoku_launcher()
            except Exception as e:
                problems.append(f"- YomiToku CLI の起動方法を解決できません: {e}")

        ok = len(problems) == 0
        msg = "\n".join(problems) if problems else "OK"
        if return_message:
            return ok, msg
        return ok, ""

    def check_cuda_available(self) -> tuple[bool, str]:
        try:
            import torch  # type: ignore
        except Exception as e:
            return False, f"PyTorch の読み込みに失敗しました: {e}"
        try:
            if bool(torch.cuda.is_available()):
                return True, "OK"
            return False, "torch.cuda.is_available() が False です（CUDA対応PyTorch/GPUドライバ未整備の可能性）"
        except Exception as e:
            return False, f"CUDA 利用可否の確認に失敗しました: {e}"

    def find_local_yomitoku_script(self) -> Optional[list[str]]:
        candidates: list[Path] = []
        py = Path(sys.executable)
        # Windows: .../Python310/python.exe -> .../Python310/Scripts/yomitoku.exe
        candidates.append(py.parent / "Scripts" / "yomitoku.exe")
        candidates.append(py.parent / "Scripts" / "yomitoku")
        # venv on POSIX-like
        candidates.append(py.parent / "yomitoku")
        # one level up/bin (some layouts)
        candidates.append(py.parent.parent / "bin" / "yomitoku")
        for c in candidates:
            try:
                if c.exists() and c.is_file():
                    return [str(c)]
            except Exception:
                continue
        return None

    def resolve_yomitoku_launcher(self) -> list[str]:
        """YomiToku CLI の呼び出し方法を解決してキャッシュする。"""
        if self._yomitoku_launcher_cache is not None:
            return list(self._yomitoku_launcher_cache)

        candidates: list[list[str]] = []

        if shutil.which("yomitoku"):
            candidates.append(["yomitoku"])

        local_script = self.find_local_yomitoku_script()
        if local_script:
            candidates.append(local_script)

        candidates.append([sys.executable, "-m", "yomitoku.cli"])
        candidates.append([sys.executable, "-m", "yomitoku"])

        seen: set[tuple[str, ...]] = set()
        uniq_candidates: list[list[str]] = []
        for c in candidates:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                uniq_candidates.append(c)

        last_err = "候補なし"
        for c in uniq_candidates:
            try:
                cp = subprocess.run(
                    c + ["--help"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=8,
                )
                text = ((cp.stdout or "") + "\n" + (cp.stderr or "")).lower()
                if cp.returncode == 0 and ("yomitoku" in text or "--reading_order" in text or "-f" in text):
                    self._yomitoku_launcher_cache = list(c)
                    return list(c)
                if "no module named yomitoku.__main__" in text:
                    last_err = "yomitoku パッケージに __main__ がありません（-m yomitoku 直実行不可）"
                else:
                    last_err = (cp.stderr or cp.stdout or f"returncode={cp.returncode}").strip()[:300]
            except Exception as e:
                last_err = str(e)

        raise RuntimeError(last_err)

class BatchProcessor:
    """1件分のバッチ処理（判定→出力先→OCR→CSV行更新）をまとめるヘルパー。"""

    def __init__(
        self,
        *,
        csv_logger: CsvLogger,
        pdf_has_text_layer: Callable[[Path], bool],
        make_output_path: Callable[[Path, bool, bool], Path],
        default_output_path: Callable[[Path, bool], Path],
        run_yomitoku: Callable[..., None],
        update_progress_ui: Callable[[int, int, str], None],
        log: Callable[[str, str], None],
    ) -> None:
        self._csv_logger = csv_logger
        self._pdf_has_text_layer = pdf_has_text_layer
        self._make_output_path = make_output_path
        self._default_output_path = default_output_path
        self._run_yomitoku = run_yomitoku
        self._update_progress_ui = update_progress_ui
        self._log = log

    def process_one(
        self,
        *,
        idx: int,
        total: int,
        pdf_path: Path,
        run_config: RunConfig,
        result: ProcessResult,
    ) -> bool:
        """1件分を処理する。停止で中断した場合のみ True を返す。"""
        t0 = time.perf_counter()
        row: Dict[str, Any] = self._csv_logger.new_row(pdf_path, run_config)

        try:
            self._update_progress_ui(idx - 1, total, f"判定中: {pdf_path.name}")

            if GENERATED_INPUT_NAME_RE.search(pdf_path.stem):
                result.skipped_name_rule += 1
                row["action"] = "skip_name"
                row["detail"] = "出力ファイル名規則に一致（再処理防止）"
                self._log("INFO", f"[SKIP:name] {pdf_path.name}")
                self._update_progress_ui(idx, total, f"スキップ（出力名）: {pdf_path.name}")
                return False

            if self._pdf_has_text_layer(pdf_path):
                result.skipped_has_text += 1
                row["action"] = "skip_has_text"
                row["detail"] = "既にテキスト層あり"
                self._log("INFO", f"[SKIP:text] 既にテキスト層あり: {pdf_path.name}")
                self._update_progress_ui(idx, total, f"スキップ（OCRあり）: {pdf_path.name}")
                return False

            out_path = self._make_output_path(pdf_path, run_config.same_folder_output, run_config.overwrite)
            row["output"] = str(out_path)
            if out_path.exists() and run_config.overwrite:
                self._log("WARN", f"[OVERWRITE] {out_path.name}")
            elif out_path.name != self._default_output_path(pdf_path, run_config.same_folder_output).name:
                self._log("INFO", f"[RENAME] 同名回避のため連番保存: {out_path.name}")

            self._log("INFO", f"[OCR] {pdf_path.name} -> {out_path.name}")
            self._update_progress_ui(idx - 1, total, f"OCR実行中: {pdf_path.name}")

            self._run_yomitoku(
                in_pdf=pdf_path,
                out_pdf=out_path,
                dpi=run_config.dpi,
                device=run_config.resolved_device,
                lite=run_config.lite,
                timeout_sec=run_config.yomitoku_timeout_sec,
            )

            if out_path.exists() and out_path.stat().st_size > 0:
                result.processed += 1
                row["action"] = "ocr_done"
                row["detail"] = "OK"
                self._log("DONE", f"[DONE] {out_path}")
            else:
                raise RuntimeError("OCR 実行後に出力ファイルが確認できませんでした")

            self._update_progress_ui(idx, total, f"完了: {pdf_path.name}")
            return False

        except OCRStopRequested as e:
            stopped = True
            row["action"] = "stopped"
            row["detail"] = str(e)
            self._log("WARN", f"[STOPPED] {pdf_path.name}: {e}")
            self._update_progress_ui(idx - 1, total, f"停止: {pdf_path.name}")
            return True
        except Exception as e:
            result.errors += 1
            row["action"] = "error"
            row["detail"] = str(e)
            self._log("ERROR", f"[ERROR] {pdf_path.name}: {e}")
            self._update_progress_ui(idx, total, f"エラー: {pdf_path.name}")
            return False
        finally:
            row["seconds"] = f"{time.perf_counter() - t0:.2f}"
            self._csv_logger.append_row(row)


class BatchRunCoordinator:
    """バッチ1回分の進行管理（列挙/集計/CSV出力/最終ログ）をまとめるヘルパー。"""

    def __init__(
        self,
        *,
        collect_pdfs: Callable[[Path, bool], List[Path]],
        make_csv_log_path: Callable[[Path], Path],
        csv_logger: CsvLogger,
        batch_processor: BatchProcessor,
        is_stop_requested: Callable[[], bool],
        update_progress_ui: Callable[[int, int, str], None],
        log: Callable[[str, str], None],
    ) -> None:
        self._collect_pdfs = collect_pdfs
        self._make_csv_log_path = make_csv_log_path
        self._csv_logger = csv_logger
        self._batch_processor = batch_processor
        self._is_stop_requested = is_stop_requested
        self._update_progress_ui = update_progress_ui
        self._log = log

    def _write_csv_if_needed(
        self,
        *,
        csv_log_path: Optional[Path],
        run_config: RunConfig,
        summary: RunSummary,
    ) -> None:
        if not csv_log_path:
            return
        try:
            self._csv_logger.write(csv_log_path, run_config, summary)
            self._log("DONE", f"[LOG] {csv_log_path}")
        except Exception as e:
            self._log("ERROR", f"[LOG-ERROR] CSVログ出力に失敗: {e}")

    @staticmethod
    def _build_final_message(summary: RunSummary) -> str:
        result = summary.result
        state = "停止" if summary.was_stopped else "完了"
        return (
            f"{state}  対象:{result.total} / OCR実行:{result.processed} / "
            f"スキップ(既存テキスト):{result.skipped_has_text} / "
            f"スキップ(名前):{result.skipped_name_rule} / エラー:{result.errors}"
        )

    @staticmethod
    def _build_final_outcome(summary: RunSummary) -> BatchRunOutcome:
        result = summary.result
        if result.total == 0:
            return BatchRunOutcome(final_status="完了（対象なし）", ui_current=0, ui_total=0)
        processed_count = result.processed + result.skipped_has_text + result.skipped_name_rule + result.errors
        return BatchRunOutcome(
            final_status="停止済み" if summary.was_stopped else "完了",
            ui_current=min(processed_count, max(result.total, 1)),
            ui_total=max(result.total, 1),
        )

    @classmethod
    def _build_completion_presentation(cls, summary: RunSummary) -> BatchCompletionPresentation:
        return BatchCompletionPresentation(
            final_log_kind=("WARN" if summary.was_stopped else "DONE"),
            final_log_message=cls._build_final_message(summary),
            outcome=cls._build_final_outcome(summary),
        )

    def run(self, run_config: RunConfig) -> BatchRunOutcome:
        folder = run_config.folder
        recursive = run_config.recursive
        started_at = datetime.now()
        csv_log_path: Optional[Path] = None
        result = ProcessResult()
        was_stopped = False

        pdfs = self._collect_pdfs(folder, recursive)
        result.total = len(pdfs)
        total = result.total

        if run_config.write_csv_log:
            csv_log_path = self._make_csv_log_path(folder)
            self._log("INFO", f"CSVログ出力: {csv_log_path}")

        if total == 0:
            self._log("WARN", "PDF ファイルが見つかりませんでした。")
            summary = RunSummary(
                started_at=started_at,
                ended_at=datetime.now(),
                result=result,
                was_stopped=False,
            )
            self._write_csv_if_needed(csv_log_path=csv_log_path, run_config=run_config, summary=summary)
            return self._build_final_outcome(summary)

        self._log("INFO", f"対象PDF数: {total}")
        self._update_progress_ui(0, total, "処理中...")

        for idx, pdf_path in enumerate(pdfs, start=1):
            if self._is_stop_requested():
                was_stopped = True
                self._log("WARN", "停止要求により処理を中断しました。")
                break

            stopped_in_item = self._batch_processor.process_one(
                idx=idx,
                total=total,
                pdf_path=pdf_path,
                run_config=run_config,
                result=result,
            )
            if stopped_in_item:
                was_stopped = True
                break

        summary = RunSummary(
            started_at=started_at,
            ended_at=datetime.now(),
            result=result,
            was_stopped=was_stopped,
        )
        presentation = self._build_completion_presentation(summary)
        self._log(presentation.final_log_kind, presentation.final_log_message)
        self._write_csv_if_needed(csv_log_path=csv_log_path, run_config=run_config, summary=summary)
        return presentation.outcome


class FinishUiCoordinator:
    """終了時UI更新（進捗/ボタン/設定再有効化）をまとめるヘルパー。"""

    def __init__(
        self,
        *,
        root: tk.Tk,
        progress: Any,
        status_var: tk.StringVar,
        progress_text_var: tk.StringVar,
        btn_start: Any,
        btn_stop: Any,
        set_runtime_controls_enabled: Callable[[bool], None],
        clear_current_run_config: Callable[[], None],
    ) -> None:
        self._root = root
        self._progress = progress
        self._status_var = status_var
        self._progress_text_var = progress_text_var
        self._btn_start = btn_start
        self._btn_stop = btn_stop
        self._set_runtime_controls_enabled = set_runtime_controls_enabled
        self._clear_current_run_config = clear_current_run_config

    @staticmethod
    def _calc_progress_pct(current: int, total: int) -> float:
        total_safe = max(total, 1)
        return max(0.0, min(100.0, (current / total_safe) * 100.0))

    def apply(self, state: FinishUiState) -> None:
        pct = self._calc_progress_pct(state.current, state.total)
        self._progress.configure(value=pct)
        self._status_var.set(state.status)
        self._progress_text_var.set(f"{min(state.current, state.total)} / {state.total}")
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._set_runtime_controls_enabled(True)
        self._clear_current_run_config()

    def schedule(self, state: FinishUiState) -> None:
        self._root.after(0, lambda: self.apply(state))


class OCRStopRequested(RuntimeError):
    """停止要求によりOCR実行を中断したことを表す例外。"""


class LogQueue:
    def __init__(self) -> None:
        self.q: "queue.Queue[tuple[str, str]]" = queue.Queue()

    def put(self, kind: str, msg: str) -> None:
        self.q.put((kind, msg))

    def get_nowait(self):
        return self.q.get_nowait()


class OCRBatchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1040x760")

        self.logq = LogQueue()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_requested = False
        self._resolved_device_for_run: str = "cpu"
        self._runtime_support = RuntimeSupport(has_pypdf=(PdfReader is not None))
        self._csv_logger = CsvLogger()
        self._pdf_inspector = PdfInspector(pdf_reader_cls=PdfReader, log=self._log)
        self._output_path_resolver = OutputPathResolver()
        self._runtime_control_containers: list[Any] = []
        self._active_process: Any = None
        self._current_run_config: Optional[RunConfig] = None
        self._yomitoku_runner = YomiTokuRunner(
            resolve_launcher=self._resolve_yomitoku_launcher,
            reset_launcher_cache=self._reset_yomitoku_launcher_cache,
            log=self._log,
            is_stop_requested=self._is_stop_requested,
            terminate_process=self._terminate_active_process,
            set_active_process=self._set_active_process,
            quote_for_log=self._quote_for_log,
        )
        self._batch_processor = BatchProcessor(
            csv_logger=self._csv_logger,
            pdf_has_text_layer=self._pdf_has_text_layer,
            make_output_path=self._make_output_path,
            default_output_path=self._default_output_path,
            run_yomitoku=self._run_yomitoku,
            update_progress_ui=self._update_progress_ui,
            log=self._log,
        )
        self._batch_run_coordinator = BatchRunCoordinator(
            collect_pdfs=self._collect_pdfs,
            make_csv_log_path=self._make_csv_log_path,
            csv_logger=self._csv_logger,
            batch_processor=self._batch_processor,
            is_stop_requested=self._is_stop_requested,
            update_progress_ui=self._update_progress_ui,
            log=self._log,
        )
        self._start_run_coordinator = StartRunCoordinator(
            validate_runtime_settings=self._validate_runtime_settings,
            check_dependencies=self._check_dependencies,
            resolve_device_for_run=self._resolve_device_for_run,
            check_cuda_available=self._check_cuda_available,
        )

        # Variables
        self.folder_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.make_output_in_same_folder_var = tk.BooleanVar(value=True)
        self.write_csv_log_var = tk.BooleanVar(value=True)
        self.compute_mode_var = tk.StringVar(value="auto")  # auto / cpu / gpu(UI) -> cpu/cuda(YomiToku)
        self.lite_var = tk.BooleanVar(value=True)
        self.dpi_var = tk.StringVar(value="200")
        self.status_var = tk.StringVar(value="待機中")
        self.progress_text_var = tk.StringVar(value="0 / 0")

        self._build_ui()
        self._finish_ui_coordinator = FinishUiCoordinator(
            root=self.root,
            progress=self.progress,
            status_var=self.status_var,
            progress_text_var=self.progress_text_var,
            btn_start=self.btn_start,
            btn_stop=self.btn_stop,
            set_runtime_controls_enabled=self._set_runtime_controls_enabled,
            clear_current_run_config=self._clear_current_run_context,
        )
        self._schedule_log_pump()
        self.root.after(200, self._startup_dependency_check)

    # -------------------- UI --------------------
    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}

        frm_top = ttk.Frame(self.root)
        self._runtime_control_containers.append(frm_top)
        frm_top.pack(fill="x", **pad)

        ttk.Label(frm_top, text="対象フォルダ:").grid(row=0, column=0, sticky="w")
        ent_folder = ttk.Entry(frm_top, textvariable=self.folder_var)
        ent_folder.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(frm_top, text="参照...", command=self.on_browse_folder).grid(row=0, column=2, sticky="e")
        frm_top.columnconfigure(1, weight=1)

        frm_opts = ttk.LabelFrame(self.root, text="設定（YomiToku / 日本語・横書き想定）")
        self._runtime_control_containers.append(frm_opts)
        frm_opts.pack(fill="x", **pad)

        ttk.Checkbutton(frm_opts, text="サブフォルダも処理する", variable=self.recursive_var).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(frm_opts, text="既存の同名出力を上書きする（OFF時は連番保存）", variable=self.overwrite_var).grid(row=0, column=1, sticky="w", padx=8, pady=6, columnspan=2)

        ttk.Checkbutton(frm_opts, text="出力を元PDFと同じフォルダに保存する", variable=self.make_output_in_same_folder_var).grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(frm_opts, text="CSVログを出力する", variable=self.write_csv_log_var).grid(row=1, column=1, sticky="w", padx=8, pady=6)

        ttk.Label(frm_opts, text="実行デバイス:").grid(row=2, column=0, sticky="e", padx=(8, 4), pady=6)
        frm_device = ttk.Frame(frm_opts)
        frm_device.grid(row=2, column=1, sticky="w", padx=(0, 8), pady=6)
        ttk.Radiobutton(frm_device, text="自動", value="auto", variable=self.compute_mode_var).pack(side="left")
        ttk.Radiobutton(frm_device, text="CPU", value="cpu", variable=self.compute_mode_var).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(frm_device, text="GPU (CUDA)", value="gpu", variable=self.compute_mode_var).pack(side="left", padx=(8, 0))

        ttk.Checkbutton(frm_opts, text="軽量モデル (--lite, CPU向け推奨)", variable=self.lite_var).grid(row=2, column=2, sticky="w", padx=8, pady=6)

        ttk.Label(frm_opts, text="PDF読取DPI:").grid(row=3, column=0, sticky="e", padx=(8, 4), pady=6)
        ttk.Entry(frm_opts, width=10, textvariable=self.dpi_var).grid(row=3, column=1, sticky="w", padx=(0, 8), pady=6)
        ttk.Label(frm_opts, text="(YomiToku --dpi / 標準200)").grid(row=3, column=2, sticky="w", padx=(0, 8), pady=6)

        frm_buttons = ttk.Frame(self.root)
        frm_buttons.pack(fill="x", **pad)
        self.btn_start = ttk.Button(frm_buttons, text="実行開始", command=self.on_start)
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop = ttk.Button(frm_buttons, text="停止要求", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left")
        ttk.Button(frm_buttons, text="ログをクリア", command=self.clear_log).pack(side="left", padx=(12, 0))

        frm_progress = ttk.Frame(self.root)
        frm_progress.pack(fill="x", **pad)
        ttk.Label(frm_progress, textvariable=self.status_var).pack(side="left")
        ttk.Label(frm_progress, textvariable=self.progress_text_var).pack(side="right")

        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.progress.pack(fill="x", **pad)

        frm_log = ttk.LabelFrame(self.root, text="ログ")
        frm_log.pack(fill="both", expand=True, **pad)

        self.txt_log = tk.Text(frm_log, height=28, wrap="none")
        self.txt_log.pack(side="left", fill="both", expand=True)
        ysb = ttk.Scrollbar(frm_log, orient="vertical", command=self.txt_log.yview)
        ysb.pack(side="right", fill="y")
        xsb = ttk.Scrollbar(self.root, orient="horizontal", command=self.txt_log.xview)
        xsb.pack(fill="x", padx=8, pady=(0, 8))
        self.txt_log.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        self.txt_log.tag_configure("INFO", foreground="#000000")
        self.txt_log.tag_configure("WARN", foreground="#b36b00")
        self.txt_log.tag_configure("ERROR", foreground="#b00020")
        self.txt_log.tag_configure("DONE", foreground="#006400")

    def on_browse_folder(self) -> None:
        d = filedialog.askdirectory(title="処理対象フォルダを選択")
        if d:
            self.folder_var.set(d)

    def clear_log(self) -> None:
        self.txt_log.delete("1.0", "end")

    def on_stop(self) -> None:
        self.stop_requested = True
        self._log("WARN", "停止要求を受け付けました（可能な範囲で速やかに停止します）。")

    def on_start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "処理中です。完了または停止後に再実行してください。")
            return

        plan = self._prepare_start_run_plan()
        if plan is None:
            return

        self._begin_run(plan)

    def _prepare_start_run_plan(self) -> Optional[StartRunPlan]:
        ok, err_msg, plan = self._start_run_coordinator.prepare(
            folder_str=self.folder_var.get(),
            recursive=bool(self.recursive_var.get()),
            overwrite=bool(self.overwrite_var.get()),
            same_folder_output=bool(self.make_output_in_same_folder_var.get()),
            write_csv_log=bool(self.write_csv_log_var.get()),
            selected_mode_raw=self.compute_mode_var.get(),
            lite=bool(self.lite_var.get()),
            dpi_text=self.dpi_var.get(),
        )
        if not ok or plan is None:
            # 事前準備段階の失敗は設定/依存関係/入力系のエラーとして扱う
            messagebox.showerror(APP_TITLE, err_msg or "実行前チェックに失敗しました。")
            return None

        if plan.gpu_warning_message:
            messagebox.showwarning(APP_TITLE, plan.gpu_warning_message)

        return plan

    def _begin_run(self, plan: StartRunPlan) -> None:
        run_config = plan.run_config
        self._resolved_device_for_run = run_config.resolved_device
        self._current_run_config = run_config

        self.stop_requested = False
        self._csv_logger.reset()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._set_runtime_controls_enabled(False)
        self.progress.configure(value=0, maximum=100)
        self.status_var.set("準備中...")
        self.progress_text_var.set("0 / 0")

        for kind, msg in plan.pre_logs:
            self._log(kind, msg)
        self._log("INFO", plan.startup_log_message)

        self.worker_thread = threading.Thread(target=self._run_batch, args=(run_config,), daemon=True)
        self.worker_thread.start()

    def _schedule_log_pump(self) -> None:
        def pump() -> None:
            try:
                while True:
                    kind, msg = self.logq.get_nowait()
                    self.txt_log.insert("end", msg + "\n", kind)
                    self.txt_log.see("end")
            except queue.Empty:
                pass
            self.root.after(100, pump)

        self.root.after(100, pump)

    def _log(self, kind: str, msg: str) -> None:
        self.logq.put(kind, msg)

    def _set_runtime_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for container in self._runtime_control_containers:
            self._apply_state_recursive(container, state)

    def _apply_state_recursive(self, widget: Any, state: str) -> None:
        for child in widget.winfo_children():
            self._apply_state_recursive(child, state)
            try:
                child.configure(state=state)
            except Exception:
                pass

    def _is_stop_requested(self) -> bool:
        return bool(self.stop_requested)

    def _set_active_process(self, proc: Any) -> None:
        self._active_process = proc

    def _reset_yomitoku_launcher_cache(self) -> None:
        self._runtime_support.reset_yomitoku_launcher_cache()

    def _clear_current_run_context(self) -> None:
        self._current_run_config = None

    def _terminate_active_process(self, proc: Any) -> None:
        if proc is None:
            return
        if proc.poll() is not None:
            return

        try:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=3)
            return
        except Exception:
            pass

        try:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
            else:
                proc.kill()
        except Exception:
            pass

        try:
            proc.wait(timeout=3)
        except Exception:
            pass


    def _run_subprocess_with_polling(self, cmd: list[str], timeout_sec: int) -> tuple[int, str, str]:
        """互換用ラッパー。実体は YomiTokuRunner に委譲。"""
        return self._yomitoku_runner.run_subprocess_with_polling(cmd, timeout_sec)

    # -------------------- Validation & Dependency --------------------
    def _startup_dependency_check(self) -> None:
        ok, msg = self._check_dependencies(return_message=True)
        if not ok:
            messagebox.showwarning(
                APP_TITLE,
                "必要なライブラリ/コマンドが不足している可能性があります。\n\n"
                + msg
                + "\n\nインストール後に再度お試しください。",
            )
        else:
            self._log("INFO", "依存関係チェック: OK（YomiToku）")

    def _validate_runtime_settings(self) -> tuple[bool, str]:
        return self._runtime_support.validate_runtime_settings(self.dpi_var.get())

    def _resolve_device_for_run(self, selected_mode: str) -> str:
        m = (selected_mode or "auto").strip().lower()
        if m == "cpu":
            return "cpu"
        if m == "gpu":
            return "cuda"
        # auto
        cuda_ok, _ = self._check_cuda_available()
        return "cuda" if cuda_ok else "cpu"

    def _check_dependencies(self, return_message: bool = False) -> tuple[bool, str]:
        return self._runtime_support.check_dependencies(return_message=return_message)

    def _check_cuda_available(self) -> tuple[bool, str]:
        return self._runtime_support.check_cuda_available()

    def _find_local_yomitoku_script(self) -> Optional[list[str]]:
        return self._runtime_support.find_local_yomitoku_script()

    def _resolve_yomitoku_launcher(self) -> list[str]:
        return self._runtime_support.resolve_yomitoku_launcher()

    def _run_batch(self, run_config: RunConfig) -> None:
        outcome = self._batch_run_coordinator.run(run_config)
        self._finish_ui(outcome.final_status, outcome.ui_current, outcome.ui_total)


    def _collect_pdfs(self, folder: Path, recursive: bool) -> List[Path]:
        """互換用ラッパー。実体は PdfInspector に委譲。"""
        return self._pdf_inspector.collect_pdfs(folder=folder, recursive=recursive)


    def _pdf_has_text_layer(self, pdf_path: Path, pages_to_check: int = 5, min_chars: int = 10) -> bool:
        """互換用ラッパー。実体は PdfInspector に委譲。"""
        return self._pdf_inspector.has_text_layer(
            pdf_path=pdf_path,
            pages_to_check=pages_to_check,
            min_chars=min_chars,
        )


    def _default_output_path(self, pdf_path: Path, same_folder_output: bool) -> Path:
        """互換用ラッパー。実体は OutputPathResolver に委譲。"""
        return self._output_path_resolver.default_output_path(
            pdf_path=pdf_path,
            same_folder_output=same_folder_output,
        )


    def _make_output_path(self, pdf_path: Path, same_folder_output: bool, overwrite: bool) -> Path:
        """互換用ラッパー。実体は OutputPathResolver に委譲。"""
        return self._output_path_resolver.make_output_path(
            pdf_path=pdf_path,
            same_folder_output=same_folder_output,
            overwrite=overwrite,
        )

    def _build_yomitoku_cmd(
        self,
        in_pdf: Path,
        outdir: Path,
        dpi: int,
        device: str,
        lite: bool,
        launcher: Optional[list[str]] = None,
    ) -> list[str]:
        """互換用ラッパー。実体は YomiTokuRunner に委譲。"""
        return self._yomitoku_runner.build_cmd(
            in_pdf=in_pdf,
            outdir=outdir,
            dpi=dpi,
            device=device,
            lite=lite,
            launcher=launcher,
        )


    def _find_generated_pdf(self, outdir: Path, input_pdf: Optional[Path] = None) -> tuple[Optional[Path], str]:
        """互換用ラッパー。実体は YomiTokuRunner に委譲。"""
        return self._yomitoku_runner.find_generated_pdf(outdir=outdir, input_pdf=input_pdf)


    def _run_yomitoku(self, in_pdf: Path, out_pdf: Path, dpi: int, device: str, lite: bool, timeout_sec: int) -> None:
        """互換用ラッパー。実体は YomiTokuRunner に委譲。"""
        self._yomitoku_runner.run(
            in_pdf=in_pdf,
            out_pdf=out_pdf,
            dpi=dpi,
            device=device,
            lite=lite,
            timeout_sec=timeout_sec,
        )

    def _make_csv_log_path(self, folder: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = folder / f"ocr_batch_log_{ts}.csv"
        if not base.exists():
            return base
        for i in range(1, 1000):
            cand = folder / f"ocr_batch_log_{ts}_{i:03d}.csv"
            if not cand.exists():
                return cand
        return folder / f"ocr_batch_log_{ts}_{int(time.time())}.csv"

    def _write_csv_log(
        self,
        csv_path: Path,
        run_config: RunConfig,
        started_at: datetime,
        ended_at: datetime,
        result: ProcessResult,
        was_stopped: bool,
    ) -> None:
        """互換用ラッパー。実体は CsvLogger に委譲。"""
        summary = RunSummary(
            started_at=started_at,
            ended_at=ended_at,
            result=result,
            was_stopped=was_stopped,
        )
        self._csv_logger.write(csv_path, run_config, summary)

    @staticmethod
    def _quote_for_log(s: str) -> str:
        if not s:
            return '""'
        if re.search(r"\s", s):
            return '"' + s.replace('"', '\\"') + '"'
        return s

    # -------------------- UI updates from worker --------------------
    def _update_progress_ui(self, current: int, total: int, status: str) -> None:
        def _apply() -> None:
            total_safe = max(total, 1)
            pct = max(0, min(100, (current / total_safe) * 100.0))
            self.progress.configure(value=pct)
            self.status_var.set(status)
            self.progress_text_var.set(f"{min(current, total)} / {total}")
        self.root.after(0, _apply)

    def _finish_ui(self, status: str, current: int, total: int) -> None:
        state = FinishUiState(status=status, current=current, total=total)
        self._finish_ui_coordinator.schedule(state)


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    OCRBatchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Background runner with a UTF-8 log, cancellation and explicit exit events."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import threading


@dataclass(frozen=True)
class ProcessEvent:
    kind: str
    payload: str | int


class RunProcess:
    def __init__(self, python: Path, test_v1: Path):
        self.python = python
        self.test_v1 = test_v1
        self.events: queue.Queue[ProcessEvent] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.cancelled = False

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, arguments: list[str], output_dir: Path, config_payload: dict | None = None) -> None:
        if self.running:
            raise RuntimeError("A runner is already active")
        command = [str(self.python), "-u", "-m", "Paradigm.tta_composition.run", *arguments]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.test_v1) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUNBUFFERED"] = "1"
        output_dir.mkdir(parents=True, exist_ok=False)
        if config_payload is not None:
            (output_dir / "gui_config.json").write_text(
                json.dumps(config_payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
        log_path = output_dir / "execution.log"
        log_path.write_text("$ " + subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        self.events = queue.Queue()
        self.cancelled = False
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.test_v1,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as error:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"啟動失敗：{error}\n")
            raise
        self.reader_thread = threading.Thread(
            target=self._read_process, args=(self.process, log_path), daemon=True
        )
        self.reader_thread.start()

    def _read_process(self, process: subprocess.Popen[str], log_path: Path) -> None:
        try:
            assert process.stdout is not None
            with process.stdout, log_path.open("a", encoding="utf-8") as stream:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    self.events.put(ProcessEvent("line", line))
        except (OSError, ValueError) as error:
            self.events.put(ProcessEvent("error", str(error)))
            if process.poll() is None:
                process.terminate()
        finally:
            self.events.put(ProcessEvent("exit", process.wait()))

    def cancel(self) -> None:
        if self.running and self.process is not None:
            self.cancelled = True
            process = self.process
            try:
                process.terminate()
            except OSError:
                if process.poll() is None:
                    raise
            timer = threading.Timer(3.0, self._kill_if_running, args=(process,))
            timer.daemon = True
            timer.start()

    @staticmethod
    def _kill_if_running(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                if process.poll() is None:
                    raise

    def close(self) -> None:
        self.cancel()
        if self.process is not None:
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._kill_if_running(self.process)
                self.process.wait(timeout=2.0)
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)

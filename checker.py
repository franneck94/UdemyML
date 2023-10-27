import os
import subprocess

import nbconvert


directory = "."


def should_exclude_directory(dir_name: str) -> bool:
    exclude_dirs = [
        ".mypy",
        ".ruff_cache",
        ".vscode",
        "__pycache__",
        ".git",
        "_Start",
    ]
    return dir_name in exclude_dirs


def check_file_for_not_implemented(file_path: str) -> bool:
    try:
        with open(file_path) as file:
            file_contents = file.read()
            if (
                "pass" in file_contents
                or "NotImplementedError" in file_contents
            ):
                return True
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return False


def check_ipynb_file(file_path: str) -> None:
    try:
        exporter = nbconvert.PythonExporter()
        (python_code, _) = exporter.from_filename(file_path)

        with open("temp_script.py", "w") as temp_script:
            temp_script.write(python_code)

            with open(os.devnull, "w"):
                process = subprocess.Popen(
                    ["python", "temp_script.py"],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
            process.wait(timeout=2)
            ret_code = process.returncode
            if ret_code == 0:
                print(f"\t\t{file_path} can be executed.")
            else:
                print(f"\t\t!!! {file_path} code {ret_code} !!!")
    except subprocess.TimeoutExpired:
        process.terminate()  # pyright: ignore
        print(f"\t\t{file_path} can be executed (interrupted).")
    except subprocess.CalledProcessError:
        print(f"\t\t!!! {file_path} cannot be executed !!!")


def check_python_file(file_path: str) -> None:
    try:
        if check_file_for_not_implemented(file_path):
            print(rf"\File: {file_path} has unfinished code")
            return
        print(f"\tRunning file: {file_path}")
        with open(os.devnull, "w") as null_file:
            process = subprocess.Popen(
                ["python", file_path],
                stderr=null_file,
                stdout=null_file,
            )
            process.wait(timeout=2)
            ret_code = process.returncode
            if ret_code == 0:
                print(f"\t\t{file_path} can be executed.")
            else:
                print(f"\t\t!!! {file_path} code {ret_code} !!!")
    except subprocess.TimeoutExpired:
        process.terminate()  # pyright: ignore
        print(f"\t\t{file_path} can be executed (interrupted).")
    except subprocess.CalledProcessError:
        print(f"\t\t!!! {file_path} cannot be executed !!!")


def main() -> None:
    for root, _, files in os.walk(directory, topdown=True):
        if should_exclude_directory(os.path.basename(root)):
            continue
        num_py_files = len([file for file in files if ".py" in file])
        if num_py_files > 0:
            print(f"dir: {root}, number of python files: {num_py_files}")
        for filename in files:
            file_path = os.path.join(root, filename)
            if "checker" in file_path:
                continue
            if file_path.endswith(".py"):
                check_python_file(file_path)
            if file_path.endswith(".ipynb"):
                check_ipynb_file(file_path)


if __name__ == "__main__":
    main()

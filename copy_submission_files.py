import shutil
import os

# Define source and destination paths
SRC_DIR = os.path.join(os.path.dirname(__file__), "src", "evaluator")
RPC_DIR = os.path.join(SRC_DIR, "rpc")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "submission_template")

# List of files to copy and their destination names
FILES_TO_COPY = [
    (os.path.join(RPC_DIR, "server.py"), os.path.join(TEMPLATE_DIR, "agent_server.py")),
    (
        os.path.join(SRC_DIR, "agent_interface.py"),
        os.path.join(TEMPLATE_DIR, "agent_interface.py"),
    ),
    (os.path.join(RPC_DIR, "agent.capnp"), os.path.join(TEMPLATE_DIR, "agent.capnp")),
]


def main():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    for src, dst in FILES_TO_COPY:
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    main()

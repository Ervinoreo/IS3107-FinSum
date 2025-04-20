import subprocess

def install_package(pkg):
    print(f"\n🔄 Installing: {pkg}")
    try:
        subprocess.check_call(["pip", "install", pkg])
        print(f"✅ Successfully installed: {pkg}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install: {pkg}")

def main():
    req_file = "requirements.txt"
    try:
        with open(req_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("🚫 requirements.txt not found.")
        return

    packages = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    for pkg in packages:
        install_package(pkg)

if __name__ == "__main__":
    main()

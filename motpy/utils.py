import importlib.util


def ensure_packages_installed(packages, stop_if_some_missing: bool = True):
    some_missing = False
    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            some_missing = True
            print(f'package {package} is not installed')

    if some_missing and stop_if_some_missing:
        print('Please install required python packages to run this script')
        exit(1)

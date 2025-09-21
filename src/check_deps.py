import sys
required_packages = ['yfinance', 'pandas', 'numpy', 'matplotlib', 'requests', 'beautifulsoup4']
missing_packages = []
for package in required_packages:
    try:
        __import__(package if package != 'beautifulsoup4' else 'bs4')
    except ImportError:
        missing_packages.append(package)
if missing_packages:
    print('Missing packages: ' + ', '.join(missing_packages))
    sys.exit(1)
else:
    print('All dependencies are installed.')
    sys.exit(0)

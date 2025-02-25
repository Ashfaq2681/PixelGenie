# PixelGenie - AI-powered Coding Assistant

## Overview
PixelGenie is a lightweight AI-powered assistant that helps with coding, debugging, optimization, and best practices. It is designed to work via a command-line interface (CLI) or programmatically in Python scripts.

## Features
- 📜 **Generate AI-powered responses** for coding-related queries
- 💻 **Explain and optimize code** in various languages
- ⚡ **Execute Python and Bash scripts** directly
- 🚀 **Suggest best practices** for software development
- 🖥 **Interactive CLI** for real-time interaction

## Installation
### Prerequisites
- Python 3.8+
- `pip` (Python package manager)
- CUDA-compatible GPU (optional for faster inference)

### Install Dependencies
```sh
pip install torch transformers
```

## Usage
### 1️⃣ Interactive CLI Mode
Run the assistant in CLI mode:
```sh
python pixelgenie.py --mode cli
```
Type your questions and get responses in real time. Type `exit` to quit.

### 2️⃣ Explain Code
To get a simple explanation of a code snippet:
```sh
python pixelgenie.py --mode explain --code "print('Hello World!')"
```

### 3️⃣ Optimize Code
To get suggestions for improving code performance:
```sh
python pixelgenie.py --mode optimize --code "for i in range(1000): print(i)"
```

### 4️⃣ Best Practices
To get best practices for a specific topic:
```sh
python pixelgenie.py --mode best-practices --topic "writing clean code"
```

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Added new feature"`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request.

## License
PixelGenie is released under the MIT License.

---
💡 **Developed with ❤️ for pixelgenie, by blucomtechnologies.com**


import json, random

MALWARE_TYPES = ["ransomware", "spyware", "trojan", "keylogger", "data-stealer"]

# Benign clean Python templates
benign_templates = [
    "def add(a,b):\n    return a+b\n\nprint('Result:', add(10,20))",
    "for i in range(5):\n    print(i)",
    "name = 'Alice'\nprint('Welcome', name)",
    "nums = [1,2,3,4]\nprint(sum(nums))",
    "import datetime\nprint(datetime.datetime.now())"
]

# Malware clean Python templates with type
malware_templates = [
    ("import requests\nurl='http://steal.info/data'\nresponse=requests.get(url)\nopen('/tmp/data.txt','w').write(response.text)", "data-stealer"),
    ("import os\nos.remove('/tmp/secret.txt')", "trojan"),
    ("import winreg\nkey=winreg.HKEY_CURRENT_USER\nwinreg.SetValue(key,'Software\\Microsoft\\Windows\\Run',winreg.REG_SZ,'malware.exe')", "trojan"),
    ("from Crypto.Cipher import AES\nkey=b'secretkey1234567'\ncipher=AES.new(key,AES.MODE_ECB)\nenc=cipher.encrypt(b'datadatadatadata')\nprint('Encrypted:',enc)", "ransomware"),
    ("import pynput\nfrom pynput import keyboard\nlog=[]\ndef on_press(key):\n    log.append(str(key))\nlistener=keyboard.Listener(on_press=on_press)\nlistener.start()", "keylogger")
]

def obfuscate_python(code: str) -> str:
    obf = code
    # Rename identifiers
    obf = obf.replace("print", f"__pr_{random.randint(1000,9999)}")
    obf = obf.replace("url", f"__u_{random.randint(1000,9999)}")
    obf = obf.replace("response", f"__res_{random.randint(1000,9999)}")
    # Add junk
    obf = "tmp=" + str(random.randint(10,99)) + "-" + str(random.randint(1,9)) + "+" + str(random.randint(1,9)) + "\n" + obf
    # Break strings
    obf = obf.replace("http", "'ht'+'tp'")
    obf = obf.replace("Alice", "'Al'+'ice'")
    return obf

def generate_dataset(n=5000, outfile="python_deobf_dataset.jsonl"):
    samples = []

    # 300 benign
    for i in range(1000):
        clean = random.choice(benign_templates)
        obf = obfuscate_python(clean)
        samples.append({
            "id": f"py_benign_{i:04d}",
            "obfuscated": obf,
            "deobfuscated": clean,
            "label": "benign"
        })

    # 300 malware
    for i in range(1000):
        clean, mtype = random.choice(malware_templates)
        obf = obfuscate_python(clean)
        samples.append({
            "id": f"py_malware_{i:04d}",
            "obfuscated": obf,
            "deobfuscated": clean,
            "label": "malware",
            "malware_type": mtype
        })

    # 200 benign PowerShell-like (but Python print with URLs)
    for i in range(1000):
        url = "http://benign.example/test"
        obf = url.replace("http", "'ht'+'tp'")
        samples.append({
            "id": f"ps_benign_{i:04d}",
            "obfuscated": f"u='{obf}'\n__pr_{random.randint(1000,9999)}(u)",
            "deobfuscated": f"url = '{url}'\nprint(url)",
            "label": "benign"
        })

    # 200 malware PowerShell-like (URL downloaders)
    malware_urls = [
        ("https://evil.com/path","trojan"),
        ("http://baddomain.org/file.exe","trojan"),
        ("http://malware.net/dl","trojan"),
        ("https://ransom.me/pay","ransomware"),
        ("https://steal.info/data","data-stealer")
    ]
    for i in range(1000):
        url, mtype = random.choice(malware_urls)
        obf = url.replace("http","'ht'+'tp'").replace("https","'htt'+'ps'")
        samples.append({
            "id": f"ps_malware_{i:04d}",
            "obfuscated": f"u='{obf}'\n__pr_{random.randint(1000,9999)}(u)",
            "deobfuscated": f"url = '{url}'\nprint(url)",
            "label": "malware",
            "malware_type": mtype
        })

    random.shuffle(samples)
    with open(outfile,"w",encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s)+"\n")
    print(f"✅ Dataset saved: {outfile} with {len(samples)} samples")

if __name__ == "__main__":
    generate_dataset()

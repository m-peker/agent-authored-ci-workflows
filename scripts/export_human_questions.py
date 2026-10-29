"""Export the human-validation items (same order and blinding as validation.html) as JSON + a .txt answer template.

Outputs: data/annotation/human/questions.json, data/annotation/human/answers_template.txt
Answer format (one line per item): `<id>: <label>`, labels TP / FP / UNSURE for F-items and A / B / TIE for P-items;
an optional note may follow after ` | `.
"""
import json
import re

D = "data/annotation/human"
GUIDE = {
    "general": "Kimin yazdigi gizlidir; tahmin etmeye calismayin. Kararlar sizin kendi yarginiz olmali "
               "(LLM/sohbet botu kullanmayin; aksi halde insan dogrulamasi gecersiz olur).",
    "finding_breaking": "TP: is akisinin basarisiz olmasina, yanlis calismasina ya da GitHub tarafindan reddedilmesine "
                        "yol acar. FP: yalnizca stil, eski surum, linter sinirlamasi ya da yapi aslinda gecerli.",
    "finding_security_high": "TP: gercekci bir saldirgan istismar edebilir (PR/issue metni run: icinde aciliyor; "
                             "pull_request_target PR kodunu calistiriyor; yazma izni/secret guvenilmeyen koda acik). "
                             "FP: girdi guvenilir, tetikleyiciye disaridan ulasilamaz ya da risk kuramsal.",
    "pair": "Ayni istek icin iki is akisi. Hangisi istegi bu depo icin calisan bir CI/CD is akisi olarak daha iyi "
            "karsiliyor? Bicim, yorum ve uzunlugu dikkate almayin. Belirgin fark yoksa TIE.",
    "answer_format": "answers_template.txt dosyasinda her satira etiketi yazin, ornek 'F-A072: TP' veya "
                     "'P-T0211: B'. Istege bagli not: 'F-A072: FP | gecerli sozdizimi'.",
}


def numbered(text, hl=None):
    return [f"{'>>' if i + 1 == hl else '  '}{i + 1:4d} | {l}" for i, l in enumerate(text.splitlines())]


def main():
    html = open(f"{D}/validation.html", encoding="utf-8").read()
    raw = re.search(r"const ITEMS = (.*?);\r?\nconst KEY", html, re.S).group(1).replace("<\/", "</")
    items = json.loads(raw)
    out = []
    for n, it in enumerate(items, 1):
        if it["kind"] == "finding":
            out.append({"no": n, "id": it["id"], "type": "finding", "category": it["category"],
                        "question": "Bu bulgu gercek bir sorun mu? (TP / FP / UNSURE)",
                        "tool_rule": f"{it['tool']}:{it['rule']}", "message": it["message"],
                        "reported_line": it["line"], "file": it["file"],
                        "code_lines": numbered(it["code"], it["line"])})
        else:
            out.append({"no": n, "id": it["id"], "type": "pair",
                        "question": "Hangi is akisi istegi daha iyi karsiliyor? (A / B / TIE)",
                        "repo": it["repo"], "language": it["language"], "path": it["path"],
                        "request": it["request"], "workflow_A": numbered(it["a"]), "workflow_B": numbered(it["b"])})
    json.dump({"guide": GUIDE, "n_items": len(out), "items": out},
              open(f"{D}/questions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(f"{D}/answers_template.txt", "w", encoding="utf-8") as fh:
        fh.write("# Degerlendirici: \n# Format: <id>: <etiket> [| not]   F-: TP/FP/UNSURE   P-: A/B/TIE\n")
        for o in out:
            fh.write(f"{o['id']}: \n")
    print(len(out), "items")


if __name__ == "__main__":
    main()

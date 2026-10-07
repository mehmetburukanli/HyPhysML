"""Creates the clean (unmarked) LaTeX sources from the marked revision:
removes \\rev{...} wrappers, {\\color{blue} ...} groups and \\color{blue}
switches, using brace matching so that the content is left untouched."""
import os, re, shutil

D = "revision3_latex"
BS = "\\"


def strip_rev(s):
    """Replace every \\rev{X} by X (brace-matched, nested-safe)."""
    key = BS + "rev{"
    out, i = [], 0
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:]); break
        out.append(s[i:j])
        k, depth = j + len(key), 1
        while depth:
            c = s[k]
            if c == BS:            # skip escaped characters such as \{ or \}
                k += 2; continue
            depth += (c == "{") - (c == "}")
            k += 1
        out.append(strip_rev(s[j + len(key):k - 1]))
        i = k
    return "".join(out)


def clean(s):
    s = strip_rev(s)
    s = s.replace("{" + BS + "color{blue}\n", "{\n").replace("{" + BS + "color{blue} ", "{")
    s = s.replace(BS + "color{blue}", "")
    s = re.sub(r"\\newcommand\{\\rev\}\[1\]\{\{#1\}\}\n", "", s)
    return s


# main manuscript
m = open(os.path.join(D, "main.tex"), encoding="utf-8").read()
m = m.replace("%% HyPhysML --- Scientific Reports, REVISED manuscript (second round).",
              "%% HyPhysML --- Scientific Reports, revised manuscript (second round), CLEAN version.")
m = re.sub(r"%%\n%% Text that is new or changed.*?revised text only\.\n", "", m, flags=re.S)
m = m.replace("%% ---- revision marking ----------------------------------------------------\n", "")
m = clean(m)
assert BS + "rev{" not in m and "color{blue}" not in m
open(os.path.join(D, "main_clean.tex"), "w", encoding="utf-8").write(m)

# supplementary information
s = open(os.path.join(D, "supplementary.tex"), encoding="utf-8").read()
s = s.replace("%% New or changed material is printed in BLUE.\n", "")
s = s.replace(" Items are numbered in the order of their first\ncitation in the main text; new or changed material is printed in blue.}",
              " Items are numbered in the order of their first\ncitation in the main text.}")
s = s.replace("supp_tables/", "supp_tables_clean/")
s = clean(s)
assert BS + "rev{" not in s and "color{blue}" not in s
open(os.path.join(D, "supplementary_clean.tex"), "w", encoding="utf-8").write(s)

# supplementary tables without colour
src, dst = os.path.join(D, "supp_tables"), os.path.join(D, "supp_tables_clean")
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    t = open(os.path.join(src, f), encoding="utf-8").read()
    t = t.replace("{" + BS + "color{blue}\n" + BS + "begin{longtable}", "{" + BS + "begin{longtable}")
    t = t.replace(BS + "color{blue}", "")
    open(os.path.join(dst, f), "w", encoding="utf-8").write(t)
print("clean sources written")

#===============================================================================
# Copyright Contributors to the oneDAL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

"""T0 static tier: check every mechanically checkable claim in the agent guidance against the tree. No LLM, $0.

Extracts links, paths, Bazel labels, bazel/make/cmake commands, Starlark macro calls and code identifiers from
the guidance files (AGENTS.md, .github/instructions, CONTRIBUTING.md) and checks each against the same tree.
Status per claim: ok / fail / unresolved. 'unresolved' is not a failure; it needs a human look.
Usage: run.py static [--tree DIR] [--out claims.jsonl]
"""
import fnmatch
import glob as globmod
import json
import os
import re
import shlex
import subprocess
from collections import Counter

TREE = None
recs = []


def ex(p):  # exists at tree
    return os.path.exists(os.path.join(TREE, p.lstrip("/")))


def read(p):
    try:
        return open(os.path.join(TREE, p), errors="replace").read()
    except FileNotFoundError:
        return ""


def load_facts():
    """Repo facts the claims are checked against: .bazelrc configs, make targets, Starlark macro signatures."""
    global GUIDE, configs, flag_alias, makefile, make_targets, plats, compilers, bzl_defs
    GUIDE = sorted(set(
        [p for p in subprocess.check_output(["git", "-C", TREE, "ls-files"], text=True).split()
         if p.endswith("AGENTS.md") or p.startswith(".github/instructions/")
         or p in (".github/copilot-instructions.md", "CONTRIBUTING.md")]))
    bazelrc = read(".bazelrc")
    configs = set(re.findall(r"^(?:build|test|run|common):([\w-]+)", bazelrc, re.M))
    flag_alias = set(re.findall(r"--flag_alias=([\w-]+)=", bazelrc))
    makefile = read("makefile")
    make_targets = set(re.findall(r"^([A-Za-z_][\w.]*)\s*:(?!=)", makefile, re.M))
    for mk in globmod.glob(os.path.join(TREE, "dev/make/**/*.mk"), recursive=True):
        make_targets |= set(re.findall(r"^([A-Za-z_][\w.]*)\s*:(?!=)", open(mk, errors="replace").read(), re.M))
    plats = {os.path.basename(p)[:-3] for p in globmod.glob(os.path.join(TREE, "dev/make/function_definitions/*.mk"))}
    compilers = {os.path.basename(p).split(".")[0]
                 for p in globmod.glob(os.path.join(TREE, "dev/make/compiler_definitions/*.mk"))}
    bzl_defs = {}
    for f in ["dev/bazel/dal.bzl", "dev/bazel/daal.bzl", "dev/bazel/cc.bzl"]:
        src = read(f)
        for m in re.finditer(r"^def (\w+)\((.*?)\):", src, re.M | re.S):
            params = [x.strip().split("=")[0].strip() for x in m.group(2).split(",") if x.strip()]
            bzl_defs[m.group(1)] = {"file": f, "params": params}


NATIVE = {"load", "package", "glob", "select", "cc_library", "cc_test", "cc_binary", "filegroup",
          "bazel_dep", "http_archive", "exports_files", "genrule"}


def build_names(pkg):
    for b in ("BUILD", "BUILD.bazel"):
        if ex(os.path.join(pkg, b)):
            s = read(os.path.join(pkg, b))
            return set(re.findall(r'name\s*=\s*"([^"]+)"', s)), True
    return set(), False


def ext_repo_names(repo):
    names = set()
    for p in globmod.glob(os.path.join(TREE, "dev/bazel/deps", "*.BUILD")):
        if os.path.basename(p).startswith(repo):
            names |= set(re.findall(r'name\s*=\s*"([^"]+)"', open(p).read()))
    for p in globmod.glob(os.path.join(TREE, "dev/bazel/deps", "*.bzl")):
        if os.path.basename(p).startswith(repo):
            names |= set(re.findall(r'name\s*=\s*"([^"]+)"', open(p).read()))
    return names


# identifiers corpus: all non-markdown tracked text in cpp/, dev/, makefile*, examples, .ci
def grep_word(tok):
    r = subprocess.run(["git", "-C", TREE, "grep", "-F", "-w", "-q", "-I", tok, "--", ".", ":!*.md", ":!*.rst"],
                       capture_output=True)
    return r.returncode == 0


def find_name(pattern, limit=3):
    """First `limit` paths ("./a/b") whose basename matches pattern, like `find . -path ./.git -prune -o -name P`,
    in sorted order so the result does not depend on the filesystem."""
    hits = []

    def walk(d, rel):
        for e in sorted(os.scandir(d), key=lambda e: e.name):
            r = f"{rel}/{e.name}"
            if r == "./.git":
                continue
            if fnmatch.fnmatchcase(e.name, pattern):
                hits.append(r)
                if len(hits) >= limit:
                    return True
            if e.is_dir(follow_symlinks=False) and walk(e.path, r):
                return True
        return False

    walk(TREE, ".")
    return hits


# ---------- extraction ----------
def add(f, line, kind, claim, status, why, ctx=""):
    recs.append(dict(file=f, line=line, kind=kind, claim=claim, status=status, why=why, ctx=ctx[:200]))


PATHLIKE = re.compile(r"^(?:\.{0,2}/)?[\w.@\-*<>]+(?:/[\w.@\-*<>{}]*)+$|^[\w\-*]+\.(?:h|hpp|cpp|i|bzl|bazel|md|rst|yml|"
                      r"yaml|mk|sh|bat|txt|ver|json|py|cmake|in|cfg|toml)$")
ROOTFILES = {"makefile", "makefile.ver", "MODULE.bazel", ".bazelrc", ".bazelversion", "INSTALL.md",
             "CONTRIBUTING.md", "BUILD", ".clang-format", ".editorconfig", ".pre-commit-config.yaml",
             "CMakeLists.txt", "WORKSPACE", "renovate.json"}


def resolve_glob(d, p0):
    pat = re.sub(r"<[^>]+>", "*", p0).replace("{", "*").replace("}", "")
    for c in {pat.lstrip("/"), os.path.normpath(os.path.join(d, pat))}:
        hits = globmod.glob(os.path.join(TREE, c), recursive=True)
        if hits:
            return ("ok", f"glob {c} -> {len(hits)} hits")
    hits = find_name(os.path.basename(pat))  # filename pattern anywhere
    if hits:
        return ("ok", f"name pattern -> e.g. {hits[0]}")
    return ("fail", f"glob {pat} -> 0 hits")


def resolve_path(f, p):
    p0 = p.strip().rstrip(".,:;")
    if p0.startswith(("http", "$", "~", "-")) or "=" in p0:
        return None
    d = os.path.dirname(f)
    cands = [p0.lstrip("/"), os.path.normpath(os.path.join(d, p0))]
    # tree-relative (e.g. `src/services/...` inside cpp/daal)
    for anc in [d] + [os.path.dirname(d)]:
        if anc:
            cands.append(os.path.normpath(os.path.join(anc, p0)))
    for tree_root in ("cpp/daal", "cpp/oneapi", "cpp"):
        cands.append(os.path.join(tree_root, p0))
    if any(c in p0 for c in "*<>{"):
        return resolve_glob(d, p0)
    for c in cands:
        if ex(c):
            return ("ok", c)
    if "/" not in p0.strip("/"):
        hits = subprocess.run(["git", "-C", TREE, "ls-files", "--", f"*{p0}"], capture_output=True,
                              text=True).stdout.split()
        hits = [h for h in hits if os.path.basename(h) == p0]
        if hits:
            return ("ok_basename", f"bare basename; {len(hits)} match(es), e.g. {hits[0]}")
    return ("unresolved", "tried " + ", ".join(dict.fromkeys(cands)))


def check_local_label(f, i, lab, pkg, tgt, ctx):
    names, has = build_names(pkg)
    if not has:
        add(f, i, "bazel_label", lab, "fail", f"no BUILD in package '{pkg}'", ctx)
        return
    t = tgt or os.path.basename(pkg)
    if t in names:
        add(f, i, "bazel_label", lab, "ok", f"name=\"{t}\" in {pkg}/BUILD", ctx)
    elif t.endswith(".bzl") or t.endswith(".BUILD") or ex(os.path.join(pkg, t)):
        add(f, i, "bazel_label", lab, "ok" if ex(os.path.join(pkg, t)) else "fail", "file label", ctx)
    else:
        add(f, i, "bazel_label", lab, "fail",
            f"no name=\"{t}\" in {pkg}/BUILD (macro-generated names need manual check)", ctx)


def check_label(f, i, lab, ctx):
    m = re.match(r"^(@[\w.-]+)?//([\w/.\-]*)(?::([\w.\-+/]+))?$", lab)
    if not m:
        return
    repo, pkg, tgt = m.groups()
    if repo is None and pkg in ("visibility", "conditions"):
        return
    if pkg.endswith("...") or lab.endswith("//..."):
        add(f, i, "bazel_label", lab, "fail", "recursive //... pattern (unsupported per dev/bazel/README; #3780 1.3)",
            ctx)
    elif repo in (None, "@onedal"):
        check_local_label(f, i, lab, pkg, tgt, ctx)
    elif repo in ("@platforms", "@config", "@bazel_skylib", "@rules_cc"):
        add(f, i, "bazel_label", lab, "unresolved", "external/config repo; manual", ctx)
    else:
        r = repo[1:]
        names = ext_repo_names(r)
        if not names:
            add(f, i, "bazel_label", lab, "unresolved", f"no dev/bazel/deps/{r}* BUILD template found", ctx)
        else:
            found = (tgt or r) in names
            add(f, i, "bazel_label", lab, "ok" if found else "fail",
                f"{'found' if found else 'missing'} in dev/bazel/deps/{r}*", ctx)


def check_bazel_cmd(f, i, toks, s):
    verb = toks[1]
    cfgs = [t.split("=", 1)[1] for t in toks if t.startswith("--config=")]
    for c in cfgs:
        add(f, i, "bazel_config", c, "ok" if c in configs else "fail", ".bazelrc config", s)
    for t in toks:
        if t.startswith("--") and not t.startswith("--config"):
            fl = t[2:].split("=")[0]
            if fl in flag_alias:
                add(f, i, "bazel_flag", t, "ok", "flag_alias in .bazelrc", s)
            elif fl != "expunge":
                add(f, i, "bazel_flag", t, "unresolved", "not a .bazelrc alias; may be a native flag", s)
    if verb in ("test", "run") and not cfgs:
        add(f, i, "bazel_cmd", s, "fail",
            f"bazel {verb} without --config (defaults to all tests incl. DPC++; dev/bazel/README)", s)
    for t in toks:
        if "//" in t:
            check_label(f, i, t, s)


def check_make_cmd(f, i, toks, s):
    for t in toks[1:]:
        if t in ("-f", "makefile") or t.startswith("-"):
            continue
        if "=" not in t:
            add(f, i, "make_target", t, "ok" if t in make_targets else "fail", "target in makefile/dev/make", s)
            continue
        k, v = t.split("=", 1)
        v = v.strip('"')
        if k == "PLAT":
            add(f, i, "make_var", t, "ok" if v in plats else "fail", f"PLAT values: {sorted(plats)}", s)
        elif k == "COMPILER":
            add(f, i, "make_var", t, "ok" if v in compilers else "fail", f"compilers: {sorted(compilers)}", s)
        else:
            add(f, i, "make_var", t, "ok" if re.search(rf"\b{k}\b", makefile) else "fail",
                "var referenced in makefile", s)


def check_cmake_cmd(f, i, toks, s):
    src = toks[toks.index("-S") + 1]
    add(f, i, "cmake_src", s, "unresolved", f"-S {src}: depends on cwd; see surrounding lines", s)
    for t in toks:
        if t.startswith("-D"):
            v = t[2:].split("=")[0]
            hit = subprocess.run(["git", "-C", TREE, "grep", "-q", "-w", v, "--", "*.txt", "*.cmake", "*.in"])
            add(f, i, "cmake_var", v, "ok" if hit.returncode == 0 else "fail", "referenced in CMake sources", s)


def check_cmd(f, i, line):
    s = line.strip().strip("`").strip()
    s = re.sub(r"\s+#.*$", "", s)
    if line.strip().startswith("`") and line.strip().endswith("`") and len(line.strip()) > 2:
        add(f, i, "shell_fence", line.strip(), "fail",
            "command wrapped in backticks inside a code fence -> command substitution if pasted (#3780 1.7)", line)
    try:
        toks = shlex.split(s)
    except ValueError:
        toks = s.split()
    if not toks:
        return
    if toks[0] == "bazel" and len(toks) > 2:  # 2 tokens is a prose mention like `bazel test`
        check_bazel_cmd(f, i, toks, s)
    if toks[0] == "make":
        check_make_cmd(f, i, toks, s)
    if toks[0] == "cmake" and "-S" in toks:
        check_cmake_cmd(f, i, toks, s)


def call_kwargs(code, m):
    """Top-level (4-space indented) keyword arguments of the call whose `name(` matched at m."""
    depth, j = 0, m.end() - 1
    k = j
    while k < len(code):
        depth += (code[k] == "(") - (code[k] == ")")
        if depth == 0:
            break
        k += 1
    body = code[j + 1:k]
    return re.findall(r"^\s{4}(\w+)\s*=", body, re.M), body


def check_starlark_block(f, fence, block):
    code = "\n".join(block)
    for m in re.finditer(r"^(\w+)\(", code, re.M):
        name = m.group(1)
        ln_no = fence + code[:m.start()].count("\n") + 1
        snippet = code[m.start():m.start() + 80]
        kws, body = call_kwargs(code, m)
        if name in bzl_defs:
            ps = bzl_defs[name]["params"]
            has_kw = any(p.startswith("**") for p in ps)
            add(f, ln_no, "starlark_macro", name, "ok", f"defined in {bzl_defs[name]['file']}", snippet)
            for x in kws:
                if x not in ps:
                    add(f, ln_no, "starlark_kwarg", f"{name}({x}=)", "unresolved" if has_kw else "fail",
                        "passed via **kwargs; manual" if has_kw else f"not a parameter of {name}", body[:120])
        elif name in ("cc_library", "cc_test"):
            add(f, ln_no, "starlark_macro", name, "fail",
                "bare cc_library/cc_test: 0 first-party BUILD files use it (#3780 1.1)", snippet)
        elif name not in NATIVE:
            add(f, ln_no, "starlark_macro", name, "fail", "not defined in dev/bazel/{dal,daal,cc}.bzl nor native",
                snippet)
    for lab in re.findall(r'"((?:@[\w.-]+)?//[^"]*)"', code):
        check_label(f, fence, lab, lab)


def check_links(f, i, ln):
    for tgt in re.findall(r"\]\(([^)\s]+)\)", ln):
        if tgt.startswith("#"):
            continue
        m = re.match(r"https://github.com/uxlfoundation/oneDAL/blob/[^/]+/(.+)", tgt)
        if m:
            tgt = "/" + m.group(1)
        elif tgt.startswith("http"):
            continue
        tgt = tgt.split("#")[0]
        p = tgt.lstrip("/") if tgt.startswith("/") else os.path.normpath(os.path.join(os.path.dirname(f), tgt))
        add(f, i, "link", tgt, "ok" if ex(p) else "fail", f"resolves to {p}", ln)


IDENT = re.compile(r"^[A-Za-z_][\w:]*(::\w+)+$|^[A-Z][A-Z0-9_]{3,}$|^[a-z]+_[a-z_]+$|^[A-Z]\w*[a-z]\w*[A-Z]\w*$")


def check_spans(f, i, ln):
    for span in re.findall(r"`([^`]+)`", ln):
        sp = span.strip()
        if sp.startswith(("bazel ", "make ", "cmake ")) or re.match(r"^[A-Z_]+=\S+ ", sp):
            check_cmd(f, i, sp)
            continue
        for lab in re.findall(r"(?:@[\w.-]+)?//[\w/.\-]*(?::[\w.\-+]+)?", sp):
            check_label(f, i, lab, sp)
        if "//" in sp:
            continue
        if PATHLIKE.match(sp) or sp in ROOTFILES:
            r = resolve_path(f, sp)
            if r:
                add(f, i, "path", sp, r[0], r[1], ln)
        elif IDENT.match(sp):
            tok = sp.split("::")[-1]
            found = grep_word(tok)
            add(f, i, "identifier", sp, "ok" if found else "fail",
                f"`{tok}` {'found' if found else 'not found'} in non-doc tracked files", ln)


def extract_file(f):
    fence, fence_lang, block = None, "", []
    for i, ln in enumerate(read(f).splitlines(), 1):
        st = ln.strip()
        if st.startswith("```"):
            if fence is None:
                fence, fence_lang, block = i, st[3:].strip().lower(), []
            else:
                if fence_lang in ("python", "starlark", "bzl", "bazel"):
                    check_starlark_block(f, fence, block)
                fence = None
        elif fence is not None:
            block.append(ln)
            if fence_lang in ("bash", "sh", "shell", ""):
                check_cmd(f, i, ln)
        else:
            check_links(f, i, ln)
            check_spans(f, i, ln)


def extract():
    for f in GUIDE:
        extract_file(f)


def main(tree, out=None):

    global TREE
    TREE = str(tree)
    recs.clear()
    load_facts()
    extract()
    if out:
        with open(out, "w") as o:
            for r in recs:
                o.write(json.dumps(r) + "\n")
    print(len(GUIDE), "guidance files;", len(recs), "claims;", dict(Counter(r["status"] for r in recs)))
    for r in recs:
        if r["status"] == "fail":
            print(f"  FAIL {r['file']}:{r['line']} [{r['kind']}] {r['claim'][:90]} -- {r['why']}")
    return recs

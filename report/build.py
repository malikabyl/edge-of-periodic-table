"""Build the Russian-language report page (report/index.html) from ../results/*.json."""
import json, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
RU = {en: ru for ru, en in json.load(open("../report/labels_ru_en.json")).items()}
def ru(o):  # translate English result labels back to the Russian page wording
    if isinstance(o, dict): return {RU.get(k, k): ru(v) for k, v in o.items()}
    if isinstance(o, list): return [ru(v) for v in o]
    return RU.get(o, o) if isinstance(o, str) else o
P = json.load(open("periodic.json"))
E = json.load(open("eval.json"))
R = json.load(open("predict.json"))
S = json.load(open("stars.json"))

QT = json.load(open("qa_test.json")); QF = json.load(open("qa_final.json"))
data = dict(
    qa=dict(n_qa=QT["n_qa"], z_range=QT["z_range"], curve=QF["curve"], targets=QF["targets"]), qa_test=QT["test"], decay=json.load(open("decay_final.json")), decay2=json.load(open("decay_final2.json")), sf2=json.load(open("sf2.json")),
    periodic=dict(elements=[[e["Z"], e["sym"], e["period"], e["block"], e["col"], e["last"]] for e in P["elements"]],
                  lengths=P["lengths"], nobles=P["nobles"], data_lengths=P["data_lengths"],
                  ie1=P["ie1"], exceptions=P["exceptions"]),
    eval=E,
    grid={k: [[c["Z"], c["N"], c["Q"], c["sQ"], c["logT"], c["logT_lo"], c["logT_hi"]] for c in v]
          for k, v in R["variants"].items()},
    measured=[[m["Z"], m["N"]] for m in R["measured"]],
    test=[[t["model"], t["Z"], t["A"], t["logT_meas"], t["logT_pred"], t["lo"], t["hi"]] for t in R["test"]],
    summary=R["summary"], vs=R["vs"], targets=R["targets"],
    stars=dict(n=S["n_stars"], n_eu=S["n_eu"], n_search=S["n_search"], labels=S["labels"],
               flagged=S["flagged_mix"], overall=S["overall_mix"], famous=S["famous"],
               scatter=[[s["Name"], s["Fe/H"], s["[Eu/Fe]"], s["label"], bool(s["is_anom"] in (True, "True"))] for s in S["scatter"]],
               thorium=[[t["Name"], t["logThEu"]] for t in S["thorium"] if t["logThEu"] is not None],
               bridge=S["bridge"]),
)
data = ru(data)
html = open("../report/index.template.html").read().replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
open("../report/index.html", "w").write(html)
print(len(html) / 1e3, "kB")

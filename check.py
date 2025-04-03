import ast

for line1, line2 in zip(open("main.log.i"), open("main.log.i.a")):
    if line1 == line2:
        continue

    l1, l2 = None, None
    if line1.startswith("AXPs: "):
        l1, l2 = line1.replace("AXPs: ", ""), line2.replace("AXPs: ", "")
    elif line1.startswith("Duals: "):
        l1, l2 = line1.replace("Duals: ", ""), line2.replace("Duals: ", "")
    elif line1.startswith("CXPs: "):
        l1, l2 = line1.replace("CXPs: ", ""), line2.replace("CXPs: ", "")
    else:
        print(line1, line2)
        break

    l1, l2 = ast.literal_eval(l1), ast.literal_eval(l2)
    s1, s2 = set(), set()
    for expl1, expl2 in zip(l1, l2):
        s1.add(frozenset(expl1))
        s2.add(frozenset(expl2))

    assert s1 - s2 == set()
    assert s2 - s1 == set()

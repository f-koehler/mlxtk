import re

REGEX_ENTRY = re.compile(r"^\s+(\d+)\s+=\s+\|\s+([\d\s]+)\>")
REGEX_NODE = re.compile(r"^\$node\s+(\d+)\s+is\s+(bosonic|fermionic)")
REGEX_LINE = re.compile(r"^-+")


def read_ns_table_ascii(path):
    tables = {}
    with open(path) as fhandle:
        for line in fhandle:
            if REGEX_LINE.match(line):
                continue

            m = REGEX_NODE.match(line)
            if m:
                index = int(m.group(1))
                tables[index] = tables.get(index, [])
                continue

            m = REGEX_ENTRY.match(line)
            if m:
                tables[index].append([int(n) for n in m.group(2).split()])
                continue

            raise RuntimeError("Unexpected line: " + line)
    return tables

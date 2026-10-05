import sys
from compiler.ttir_reader import parse_ttir, dump

ttir = open(sys.argv[1]).read()
print(dump(parse_ttir(ttir)))

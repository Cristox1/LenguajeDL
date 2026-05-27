import sys
sys.setrecursionlimit(50000)
from antlr4 import *
from PELELexer import PELELexer
from PELEParser import PELEParser
from visitorPELE import EvalVisitor

def run_code(code: str, stop_on_error: bool = False) -> None:
    input_stream = InputStream(code)
    lexer = PELELexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = PELEParser(stream)
    tree = parser.program()
    visitor = EvalVisitor()
    visitor.stop_on_error = stop_on_error
    visitor.visit(tree)

def main():
    # Leer librerias en orden de dependencia
    libs = [
        "pele_math.pele",
        "pele_numpy.pele",
        "pele_tensor.pele",
        "pele_random.pele",
        "pele_losses.pele",
        "pele_metrics.pele",
        "pele_data.pele",
        "pele_ml.pele",
        "pele_nn.pele",
        "pele_sklearn.pele",
        "pele_plot.pele",
    ]

    code_libs = ""
    for lib in libs:
        try:
            with open(lib, "r", encoding="utf-8") as f:
                code_libs += f.read() + "\n"
        except FileNotFoundError:
            pass

    # Leer el programa del usuario
    with open("programa.txt", "r", encoding="utf-8") as f:
        code_prog = f.read()

    # Combinar y ejecutar
    run_code(code_libs + "\n" + code_prog)

if __name__ == '__main__':
    main()
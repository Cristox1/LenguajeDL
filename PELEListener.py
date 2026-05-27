# Generated from PELE.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .PELEParser import PELEParser
else:
    from PELEParser import PELEParser

# This class defines a complete listener for a parse tree produced by PELEParser.
class PELEListener(ParseTreeListener):

    # Enter a parse tree produced by PELEParser#program.
    def enterProgram(self, ctx:PELEParser.ProgramContext):
        pass

    # Exit a parse tree produced by PELEParser#program.
    def exitProgram(self, ctx:PELEParser.ProgramContext):
        pass


    # Enter a parse tree produced by PELEParser#block.
    def enterBlock(self, ctx:PELEParser.BlockContext):
        pass

    # Exit a parse tree produced by PELEParser#block.
    def exitBlock(self, ctx:PELEParser.BlockContext):
        pass


    # Enter a parse tree produced by PELEParser#assignStmt.
    def enterAssignStmt(self, ctx:PELEParser.AssignStmtContext):
        pass

    # Exit a parse tree produced by PELEParser#assignStmt.
    def exitAssignStmt(self, ctx:PELEParser.AssignStmtContext):
        pass


    # Enter a parse tree produced by PELEParser#exprStmt.
    def enterExprStmt(self, ctx:PELEParser.ExprStmtContext):
        pass

    # Exit a parse tree produced by PELEParser#exprStmt.
    def exitExprStmt(self, ctx:PELEParser.ExprStmtContext):
        pass


    # Enter a parse tree produced by PELEParser#mostrarStmt.
    def enterMostrarStmt(self, ctx:PELEParser.MostrarStmtContext):
        pass

    # Exit a parse tree produced by PELEParser#mostrarStmt.
    def exitMostrarStmt(self, ctx:PELEParser.MostrarStmtContext):
        pass


    # Enter a parse tree produced by PELEParser#ifStmt.
    def enterIfStmt(self, ctx:PELEParser.IfStmtContext):
        pass

    # Exit a parse tree produced by PELEParser#ifStmt.
    def exitIfStmt(self, ctx:PELEParser.IfStmtContext):
        pass


    # Enter a parse tree produced by PELEParser#cicloWhile.
    def enterCicloWhile(self, ctx:PELEParser.CicloWhileContext):
        pass

    # Exit a parse tree produced by PELEParser#cicloWhile.
    def exitCicloWhile(self, ctx:PELEParser.CicloWhileContext):
        pass


    # Enter a parse tree produced by PELEParser#cFor.
    def enterCFor(self, ctx:PELEParser.CForContext):
        pass

    # Exit a parse tree produced by PELEParser#cFor.
    def exitCFor(self, ctx:PELEParser.CForContext):
        pass


    # Enter a parse tree produced by PELEParser#forEach.
    def enterForEach(self, ctx:PELEParser.ForEachContext):
        pass

    # Exit a parse tree produced by PELEParser#forEach.
    def exitForEach(self, ctx:PELEParser.ForEachContext):
        pass


    # Enter a parse tree produced by PELEParser#functionDeclStmt.
    def enterFunctionDeclStmt(self, ctx:PELEParser.FunctionDeclStmtContext):
        pass

    # Exit a parse tree produced by PELEParser#functionDeclStmt.
    def exitFunctionDeclStmt(self, ctx:PELEParser.FunctionDeclStmtContext):
        pass


    # Enter a parse tree produced by PELEParser#returnStmt.
    def enterReturnStmt(self, ctx:PELEParser.ReturnStmtContext):
        pass

    # Exit a parse tree produced by PELEParser#returnStmt.
    def exitReturnStmt(self, ctx:PELEParser.ReturnStmtContext):
        pass


    # Enter a parse tree produced by PELEParser#ifStatement.
    def enterIfStatement(self, ctx:PELEParser.IfStatementContext):
        pass

    # Exit a parse tree produced by PELEParser#ifStatement.
    def exitIfStatement(self, ctx:PELEParser.IfStatementContext):
        pass


    # Enter a parse tree produced by PELEParser#functionDecl.
    def enterFunctionDecl(self, ctx:PELEParser.FunctionDeclContext):
        pass

    # Exit a parse tree produced by PELEParser#functionDecl.
    def exitFunctionDecl(self, ctx:PELEParser.FunctionDeclContext):
        pass


    # Enter a parse tree produced by PELEParser#params.
    def enterParams(self, ctx:PELEParser.ParamsContext):
        pass

    # Exit a parse tree produced by PELEParser#params.
    def exitParams(self, ctx:PELEParser.ParamsContext):
        pass


    # Enter a parse tree produced by PELEParser#assignment.
    def enterAssignment(self, ctx:PELEParser.AssignmentContext):
        pass

    # Exit a parse tree produced by PELEParser#assignment.
    def exitAssignment(self, ctx:PELEParser.AssignmentContext):
        pass


    # Enter a parse tree produced by PELEParser#PostfixExpr.
    def enterPostfixExpr(self, ctx:PELEParser.PostfixExprContext):
        pass

    # Exit a parse tree produced by PELEParser#PostfixExpr.
    def exitPostfixExpr(self, ctx:PELEParser.PostfixExprContext):
        pass


    # Enter a parse tree produced by PELEParser#AndExpr.
    def enterAndExpr(self, ctx:PELEParser.AndExprContext):
        pass

    # Exit a parse tree produced by PELEParser#AndExpr.
    def exitAndExpr(self, ctx:PELEParser.AndExprContext):
        pass


    # Enter a parse tree produced by PELEParser#PowerExpr.
    def enterPowerExpr(self, ctx:PELEParser.PowerExprContext):
        pass

    # Exit a parse tree produced by PELEParser#PowerExpr.
    def exitPowerExpr(self, ctx:PELEParser.PowerExprContext):
        pass


    # Enter a parse tree produced by PELEParser#MulDivModExpr.
    def enterMulDivModExpr(self, ctx:PELEParser.MulDivModExprContext):
        pass

    # Exit a parse tree produced by PELEParser#MulDivModExpr.
    def exitMulDivModExpr(self, ctx:PELEParser.MulDivModExprContext):
        pass


    # Enter a parse tree produced by PELEParser#EqExpr.
    def enterEqExpr(self, ctx:PELEParser.EqExprContext):
        pass

    # Exit a parse tree produced by PELEParser#EqExpr.
    def exitEqExpr(self, ctx:PELEParser.EqExprContext):
        pass


    # Enter a parse tree produced by PELEParser#PipeExpr.
    def enterPipeExpr(self, ctx:PELEParser.PipeExprContext):
        pass

    # Exit a parse tree produced by PELEParser#PipeExpr.
    def exitPipeExpr(self, ctx:PELEParser.PipeExprContext):
        pass


    # Enter a parse tree produced by PELEParser#NotExpr.
    def enterNotExpr(self, ctx:PELEParser.NotExprContext):
        pass

    # Exit a parse tree produced by PELEParser#NotExpr.
    def exitNotExpr(self, ctx:PELEParser.NotExprContext):
        pass


    # Enter a parse tree produced by PELEParser#RelationalExpr.
    def enterRelationalExpr(self, ctx:PELEParser.RelationalExprContext):
        pass

    # Exit a parse tree produced by PELEParser#RelationalExpr.
    def exitRelationalExpr(self, ctx:PELEParser.RelationalExprContext):
        pass


    # Enter a parse tree produced by PELEParser#AddSubExpr.
    def enterAddSubExpr(self, ctx:PELEParser.AddSubExprContext):
        pass

    # Exit a parse tree produced by PELEParser#AddSubExpr.
    def exitAddSubExpr(self, ctx:PELEParser.AddSubExprContext):
        pass


    # Enter a parse tree produced by PELEParser#OrExpr.
    def enterOrExpr(self, ctx:PELEParser.OrExprContext):
        pass

    # Exit a parse tree produced by PELEParser#OrExpr.
    def exitOrExpr(self, ctx:PELEParser.OrExprContext):
        pass


    # Enter a parse tree produced by PELEParser#UnaryMinusExpr.
    def enterUnaryMinusExpr(self, ctx:PELEParser.UnaryMinusExprContext):
        pass

    # Exit a parse tree produced by PELEParser#UnaryMinusExpr.
    def exitUnaryMinusExpr(self, ctx:PELEParser.UnaryMinusExprContext):
        pass


    # Enter a parse tree produced by PELEParser#AtomExpr.
    def enterAtomExpr(self, ctx:PELEParser.AtomExprContext):
        pass

    # Exit a parse tree produced by PELEParser#AtomExpr.
    def exitAtomExpr(self, ctx:PELEParser.AtomExprContext):
        pass


    # Enter a parse tree produced by PELEParser#IndexExpr.
    def enterIndexExpr(self, ctx:PELEParser.IndexExprContext):
        pass

    # Exit a parse tree produced by PELEParser#IndexExpr.
    def exitIndexExpr(self, ctx:PELEParser.IndexExprContext):
        pass


    # Enter a parse tree produced by PELEParser#MethodCallExpr.
    def enterMethodCallExpr(self, ctx:PELEParser.MethodCallExprContext):
        pass

    # Exit a parse tree produced by PELEParser#MethodCallExpr.
    def exitMethodCallExpr(self, ctx:PELEParser.MethodCallExprContext):
        pass


    # Enter a parse tree produced by PELEParser#ParensExpr.
    def enterParensExpr(self, ctx:PELEParser.ParensExprContext):
        pass

    # Exit a parse tree produced by PELEParser#ParensExpr.
    def exitParensExpr(self, ctx:PELEParser.ParensExprContext):
        pass


    # Enter a parse tree produced by PELEParser#FuncCallExpr.
    def enterFuncCallExpr(self, ctx:PELEParser.FuncCallExprContext):
        pass

    # Exit a parse tree produced by PELEParser#FuncCallExpr.
    def exitFuncCallExpr(self, ctx:PELEParser.FuncCallExprContext):
        pass


    # Enter a parse tree produced by PELEParser#ArrayExpr.
    def enterArrayExpr(self, ctx:PELEParser.ArrayExprContext):
        pass

    # Exit a parse tree produced by PELEParser#ArrayExpr.
    def exitArrayExpr(self, ctx:PELEParser.ArrayExprContext):
        pass


    # Enter a parse tree produced by PELEParser#EmptyDictExpr.
    def enterEmptyDictExpr(self, ctx:PELEParser.EmptyDictExprContext):
        pass

    # Exit a parse tree produced by PELEParser#EmptyDictExpr.
    def exitEmptyDictExpr(self, ctx:PELEParser.EmptyDictExprContext):
        pass


    # Enter a parse tree produced by PELEParser#DictLiteralExpr.
    def enterDictLiteralExpr(self, ctx:PELEParser.DictLiteralExprContext):
        pass

    # Exit a parse tree produced by PELEParser#DictLiteralExpr.
    def exitDictLiteralExpr(self, ctx:PELEParser.DictLiteralExprContext):
        pass


    # Enter a parse tree produced by PELEParser#LambdaExpr.
    def enterLambdaExpr(self, ctx:PELEParser.LambdaExprContext):
        pass

    # Exit a parse tree produced by PELEParser#LambdaExpr.
    def exitLambdaExpr(self, ctx:PELEParser.LambdaExprContext):
        pass


    # Enter a parse tree produced by PELEParser#BoolExpr.
    def enterBoolExpr(self, ctx:PELEParser.BoolExprContext):
        pass

    # Exit a parse tree produced by PELEParser#BoolExpr.
    def exitBoolExpr(self, ctx:PELEParser.BoolExprContext):
        pass


    # Enter a parse tree produced by PELEParser#StringExpr.
    def enterStringExpr(self, ctx:PELEParser.StringExprContext):
        pass

    # Exit a parse tree produced by PELEParser#StringExpr.
    def exitStringExpr(self, ctx:PELEParser.StringExprContext):
        pass


    # Enter a parse tree produced by PELEParser#IntExpr.
    def enterIntExpr(self, ctx:PELEParser.IntExprContext):
        pass

    # Exit a parse tree produced by PELEParser#IntExpr.
    def exitIntExpr(self, ctx:PELEParser.IntExprContext):
        pass


    # Enter a parse tree produced by PELEParser#FloatExpr.
    def enterFloatExpr(self, ctx:PELEParser.FloatExprContext):
        pass

    # Exit a parse tree produced by PELEParser#FloatExpr.
    def exitFloatExpr(self, ctx:PELEParser.FloatExprContext):
        pass


    # Enter a parse tree produced by PELEParser#IdExpr.
    def enterIdExpr(self, ctx:PELEParser.IdExprContext):
        pass

    # Exit a parse tree produced by PELEParser#IdExpr.
    def exitIdExpr(self, ctx:PELEParser.IdExprContext):
        pass


    # Enter a parse tree produced by PELEParser#dictEntry.
    def enterDictEntry(self, ctx:PELEParser.DictEntryContext):
        pass

    # Exit a parse tree produced by PELEParser#dictEntry.
    def exitDictEntry(self, ctx:PELEParser.DictEntryContext):
        pass



del PELEParser
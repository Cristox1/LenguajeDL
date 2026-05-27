# Generated from PELE.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .PELEParser import PELEParser
else:
    from PELEParser import PELEParser

# This class defines a complete generic visitor for a parse tree produced by PELEParser.

class PELEVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by PELEParser#program.
    def visitProgram(self, ctx:PELEParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#block.
    def visitBlock(self, ctx:PELEParser.BlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#assignStmt.
    def visitAssignStmt(self, ctx:PELEParser.AssignStmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#exprStmt.
    def visitExprStmt(self, ctx:PELEParser.ExprStmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#mostrarStmt.
    def visitMostrarStmt(self, ctx:PELEParser.MostrarStmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#ifStmt.
    def visitIfStmt(self, ctx:PELEParser.IfStmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#cicloWhile.
    def visitCicloWhile(self, ctx:PELEParser.CicloWhileContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#cFor.
    def visitCFor(self, ctx:PELEParser.CForContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#forEach.
    def visitForEach(self, ctx:PELEParser.ForEachContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#functionDeclStmt.
    def visitFunctionDeclStmt(self, ctx:PELEParser.FunctionDeclStmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#returnStmt.
    def visitReturnStmt(self, ctx:PELEParser.ReturnStmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#ifStatement.
    def visitIfStatement(self, ctx:PELEParser.IfStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#functionDecl.
    def visitFunctionDecl(self, ctx:PELEParser.FunctionDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#params.
    def visitParams(self, ctx:PELEParser.ParamsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#assignment.
    def visitAssignment(self, ctx:PELEParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#PostfixExpr.
    def visitPostfixExpr(self, ctx:PELEParser.PostfixExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#AndExpr.
    def visitAndExpr(self, ctx:PELEParser.AndExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#PowerExpr.
    def visitPowerExpr(self, ctx:PELEParser.PowerExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#MulDivModExpr.
    def visitMulDivModExpr(self, ctx:PELEParser.MulDivModExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#EqExpr.
    def visitEqExpr(self, ctx:PELEParser.EqExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#PipeExpr.
    def visitPipeExpr(self, ctx:PELEParser.PipeExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#NotExpr.
    def visitNotExpr(self, ctx:PELEParser.NotExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#RelationalExpr.
    def visitRelationalExpr(self, ctx:PELEParser.RelationalExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#AddSubExpr.
    def visitAddSubExpr(self, ctx:PELEParser.AddSubExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#OrExpr.
    def visitOrExpr(self, ctx:PELEParser.OrExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#UnaryMinusExpr.
    def visitUnaryMinusExpr(self, ctx:PELEParser.UnaryMinusExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#AtomExpr.
    def visitAtomExpr(self, ctx:PELEParser.AtomExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#IndexExpr.
    def visitIndexExpr(self, ctx:PELEParser.IndexExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#MethodCallExpr.
    def visitMethodCallExpr(self, ctx:PELEParser.MethodCallExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#ParensExpr.
    def visitParensExpr(self, ctx:PELEParser.ParensExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#FuncCallExpr.
    def visitFuncCallExpr(self, ctx:PELEParser.FuncCallExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#ArrayExpr.
    def visitArrayExpr(self, ctx:PELEParser.ArrayExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#EmptyDictExpr.
    def visitEmptyDictExpr(self, ctx:PELEParser.EmptyDictExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#DictLiteralExpr.
    def visitDictLiteralExpr(self, ctx:PELEParser.DictLiteralExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#LambdaExpr.
    def visitLambdaExpr(self, ctx:PELEParser.LambdaExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#BoolExpr.
    def visitBoolExpr(self, ctx:PELEParser.BoolExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#StringExpr.
    def visitStringExpr(self, ctx:PELEParser.StringExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#IntExpr.
    def visitIntExpr(self, ctx:PELEParser.IntExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#FloatExpr.
    def visitFloatExpr(self, ctx:PELEParser.FloatExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#IdExpr.
    def visitIdExpr(self, ctx:PELEParser.IdExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PELEParser#dictEntry.
    def visitDictEntry(self, ctx:PELEParser.DictEntryContext):
        return self.visitChildren(ctx)



del PELEParser
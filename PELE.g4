grammar PELE;

// LÉXICOS
TRUE    : 'true' ;
FALSE   : 'false' ;
STRING  : '"' ( '\\' . | ~["\\] )* '"' ;

SI      : 'si' ;
SINO    : 'sino' ;
MOSTRAR : 'mostrar' ;
MIENTRAS: 'mientras' ;
POR     : 'por' ;
FOR     : 'for' ;
IN      : 'in' ;
FUNCION : 'funcion' ;
RETORNAR: 'retornar' ;
NOT     : 'no' ;

// Operadores multi-caracter (deben ir antes de los simples)
PIPE    : '|>' ;
AND     : '&&' ;
OR      : '||' ;
ARROW   : '->' ;
POW     : '**' ;
EQEQ    : '==' ;
NEQ     : '!=' ;
LE      : '<=' ;
GE      : '>=' ;

// Operadores simples
LT      : '<' ;
GT      : '>' ;
BACKSLASH : '\\' ;
DOT     : '.' ;
COLON   : ':' ;

// Identificador y números
ID      : [a-zA-Z_][a-zA-Z0-9_]* ;
FLOAT   : [0-9]+ '.' [0-9]+ ;
INT     : [0-9]+ ;

// Espacios y comentarios
WS      : [ \t\r\n]+ -> skip ;
COMMENT : '//' ~[\r\n]* -> skip ;

// SINTÁCTICAS
program : block EOF ;

block
    : statement*
    ;

statement
    : assignment ';'                                       # assignStmt
    | expr ';'                                             # exprStmt
    | MOSTRAR '(' expr ')' ';'                             # mostrarStmt
    | ifStatement                                          # ifStmt
    | MIENTRAS '(' expr ')' '{' block '}'                  # cicloWhile
    | POR '(' assignment ';' expr ';' assignment ')' '{' block '}'   # cFor
    | FOR '(' ID IN expr ')' '{' block '}'                 # forEach
    | functionDecl                                         # functionDeclStmt
    | RETORNAR expr ';'                                    # returnStmt
    ;

ifStatement
    : SI '(' expr ')' '{' block '}' ( SINO ifStatement | SINO '{' block '}' )?
    ;

functionDecl
    : FUNCION ID '(' params? ')' '{' block '}'
    ;

params
    : ID (',' ID)*
    ;

assignment
    : ID '=' expr
    ;

// Expresiones — de menor a mayor precedencia
expr
    : expr PIPE expr                               # PipeExpr
    | expr OR expr                                 # OrExpr
    | expr AND expr                                # AndExpr
    | NOT expr                                     # NotExpr
    | expr (EQEQ | NEQ) expr                       # EqExpr
    | expr (LT | LE | GT | GE) expr                # RelationalExpr
    | expr ('+' | '-') expr                        # AddSubExpr
    | expr ('*' | '/' | '%') expr                  # MulDivModExpr
    | expr POW expr                                # PowerExpr
    | '-' expr                                     # UnaryMinusExpr
    | postfix                                      # PostfixExpr
    ;

// Postfix — acceso por índice y llamadas a métodos
postfix
    : postfix '[' expr ']'                                     # IndexExpr
    | postfix DOT ID '(' (expr (',' expr)*)? ')'              # MethodCallExpr
    | atom                                                     # AtomExpr
    ;

atom
    : '(' expr ')'                                 # ParensExpr
    | ID '(' (expr (',' expr)*)? ')'               # FuncCallExpr
    | '[' (expr (',' expr)*)? ']'                  # ArrayExpr
    | '{' '}'                                      # EmptyDictExpr
    | '{' dictEntry (',' dictEntry)* '}'           # DictLiteralExpr
    | BACKSLASH ID ARROW expr                      # LambdaExpr
    | TRUE                                         # BoolExpr
    | FALSE                                        # BoolExpr
    | STRING                                       # StringExpr
    | INT                                          # IntExpr
    | FLOAT                                        # FloatExpr
    | ID                                           # IdExpr
    ;

dictEntry
    : (STRING | ID) COLON expr
    ;
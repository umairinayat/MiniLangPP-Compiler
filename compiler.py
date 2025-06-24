"""
Enhanced MiniLang++ Compiler Frontend with Arrays and Complete Compilation Pipeline
A complete implementation including lexical analyzer, parser, semantic analyzer,
intermediate code generator, code optimizer, and assembly code generator.

This is a modular, professional implementation of the MiniLang++ compiler.
"""

import re
import enum
from typing import List, Dict, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy

# =============================================================================
# TOKEN DEFINITIONS AND LEXICAL ANALYZER
# =============================================================================

class TokenType(enum.Enum):
    # Literals
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    STRING = "STRING"
    
    # Identifiers
    IDENTIFIER = "IDENTIFIER"
    
    # Keywords
    INT = "int"
    FLOAT_KW = "float"
    BOOL = "bool"
    STRING_KW = "string"
    VOID = "void"
    IF = "if"
    ELSE = "else"
    WHILE = "while"
    FOR = "for"
    RETURN = "return"
    TRUE = "true"
    FALSE = "false"
    BREAK = "break"
    CONTINUE = "continue"
    
    # Operators
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    ASSIGN = "="
    PLUS_ASSIGN = "+="
    MINUS_ASSIGN = "-="
    MULTIPLY_ASSIGN = "*="
    DIVIDE_ASSIGN = "/="
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_THAN = "<"
    GREATER_THAN = ">"
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"
    LOGICAL_NOT = "!"
    INCREMENT = "++"
    DECREMENT = "--"
    
    # Delimiters
    SEMICOLON = ";"
    COMMA = ","
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    LEFT_BRACE = "{"
    RIGHT_BRACE = "}"
    LEFT_BRACKET = "["
    RIGHT_BRACKET = "]"
    DOT = "."
    
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class LexicalError(Exception):
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Lexical Error at line {line}, column {column}: {message}")

class LexicalAnalyzer:
    def __init__(self, source_code: str):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.errors = []
        
        # Keywords mapping
        self.keywords = {
            'int': TokenType.INT,
            'float': TokenType.FLOAT_KW,
            'bool': TokenType.BOOL,
            'string': TokenType.STRING_KW,
            'void': TokenType.VOID,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'return': TokenType.RETURN,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE
        }
        
        # Multi-character operators
        self.multi_char_ops = {
            '==': TokenType.EQUAL,
            '!=': TokenType.NOT_EQUAL,
            '<=': TokenType.LESS_EQUAL,
            '>=': TokenType.GREATER_EQUAL,
            '&&': TokenType.LOGICAL_AND,
            '||': TokenType.LOGICAL_OR,
            '++': TokenType.INCREMENT,
            '--': TokenType.DECREMENT,
            '+=': TokenType.PLUS_ASSIGN,
            '-=': TokenType.MINUS_ASSIGN,
            '*=': TokenType.MULTIPLY_ASSIGN,
            '/=': TokenType.DIVIDE_ASSIGN
        }
        
        # Single-character operators and delimiters
        self.single_char_tokens = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO,
            '=': TokenType.ASSIGN,
            '<': TokenType.LESS_THAN,
            '>': TokenType.GREATER_THAN,
            '!': TokenType.LOGICAL_NOT,
            ';': TokenType.SEMICOLON,
            ',': TokenType.COMMA,
            '(': TokenType.LEFT_PAREN,
            ')': TokenType.RIGHT_PAREN,
            '{': TokenType.LEFT_BRACE,
            '}': TokenType.RIGHT_BRACE,
            '[': TokenType.LEFT_BRACKET,
            ']': TokenType.RIGHT_BRACKET,
            '.': TokenType.DOT
        }
    
    def current_char(self) -> str:
        if self.position >= len(self.source):
            return '\0'
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> str:
        peek_pos = self.position + offset
        if peek_pos >= len(self.source):
            return '\0'
        return self.source[peek_pos]
    
    def advance(self):
        if self.position < len(self.source) and self.source[self.position] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.position += 1
    
    def skip_whitespace(self):
        while self.current_char().isspace() and self.current_char() != '\n':
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '/' and self.peek_char() == '/':
            while self.current_char() != '\n' and self.current_char() != '\0':
                self.advance()
        elif self.current_char() == '/' and self.peek_char() == '*':
            self.advance()  # skip '/'
            self.advance()  # skip '*'
            while self.current_char() != '\0':
                if self.current_char() == '*' and self.peek_char() == '/':
                    self.advance()  # skip '*'
                    self.advance()  # skip '/'
                    break
                self.advance()
    
    def read_number(self) -> Token:
        start_line, start_col = self.line, self.column
        value = ""
        is_float = False
        
        while self.current_char().isdigit():
            value += self.current_char()
            self.advance()
        
        if self.current_char() == '.' and self.peek_char().isdigit():
            is_float = True
            value += self.current_char()
            self.advance()
            while self.current_char().isdigit():
                value += self.current_char()
                self.advance()
        
        token_type = TokenType.FLOAT if is_float else TokenType.INTEGER
        return Token(token_type, value, start_line, start_col)
    
    def read_string(self) -> Token:
        start_line, start_col = self.line, self.column
        value = ""
        self.advance()  # skip opening quote
        
        while self.current_char() != '"' and self.current_char() != '\0':
            if self.current_char() == '\\':
                self.advance()
                if self.current_char() in ['n', 't', 'r', '\\', '"']:
                    escape_chars = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"'}
                    value += escape_chars[self.current_char()]
                else:
                    value += self.current_char()
            else:
                value += self.current_char()
            self.advance()
        
        if self.current_char() == '"':
            self.advance()  # skip closing quote
        else:
            self.errors.append(LexicalError("Unterminated string literal", start_line, start_col))
        
        return Token(TokenType.STRING, value, start_line, start_col)
    
    def read_identifier(self) -> Token:
        start_line, start_col = self.line, self.column
        value = ""
        
        while (self.current_char().isalnum() or self.current_char() == '_'):
            value += self.current_char()
            self.advance()
        
        # Check if it's a keyword
        token_type = self.keywords.get(value, TokenType.IDENTIFIER)
        if token_type in [TokenType.TRUE, TokenType.FALSE]:
            token_type = TokenType.BOOLEAN
        
        return Token(token_type, value, start_line, start_col)
    
    def tokenize(self) -> List[Token]:
        while self.position < len(self.source):
            self.skip_whitespace()
            
            if self.current_char() == '\0':
                break
            
            # Handle newlines
            if self.current_char() == '\n':
                self.advance()
                continue
            
            # Handle comments
            if self.current_char() == '/' and self.peek_char() in ['/', '*']:
                self.skip_comment()
                continue
            
            start_line, start_col = self.line, self.column
            
            # String literals
            if self.current_char() == '"':
                self.tokens.append(self.read_string())
                continue
            
            # Numbers
            if self.current_char().isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Identifiers and keywords
            if self.current_char().isalpha() or self.current_char() == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Multi-character operators
            two_char = self.current_char() + self.peek_char()
            if two_char in self.multi_char_ops:
                self.advance()
                self.advance()
                self.tokens.append(Token(self.multi_char_ops[two_char], two_char, start_line, start_col))
                continue
            
            # Single-character tokens
            if self.current_char() in self.single_char_tokens:
                char = self.current_char()
                self.advance()
                self.tokens.append(Token(self.single_char_tokens[char], char, start_line, start_col))
                continue
            
            # Invalid character
            invalid_char = self.current_char()
            self.errors.append(LexicalError(f"Invalid character '{invalid_char}'", start_line, start_col))
            self.advance()
        
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return self.tokens

# =============================================================================
# ABSTRACT SYNTAX TREE NODES
# =============================================================================

class ASTNode(ABC):
    pass

@dataclass
class Program(ASTNode):
    functions: List['Function']

@dataclass
class Function(ASTNode):
    name: str
    parameters: List['Parameter']
    return_type: str
    body: List['Statement']

@dataclass
class Parameter(ASTNode):
    name: str
    type: str
    is_array: bool = False
    array_size: Optional[int] = None

@dataclass
class Statement(ASTNode):
    pass

@dataclass
class VarDeclaration(Statement):
    name: str
    type: str
    is_array: bool = False
    array_size: Optional[int] = None
    initializer: Optional['Expression'] = None

@dataclass
class Assignment(Statement):
    target: 'Expression'  # Can be identifier or array access
    value: 'Expression'

@dataclass
class IfStatement(Statement):
    condition: 'Expression'
    then_stmt: 'Statement'
    else_stmt: Optional['Statement'] = None

@dataclass
class WhileStatement(Statement):
    condition: 'Expression'
    body: 'Statement'

@dataclass
class ForStatement(Statement):
    init: Optional['Statement']
    condition: Optional['Expression']
    update: Optional['Expression']
    body: 'Statement'

@dataclass
class ReturnStatement(Statement):
    value: Optional['Expression'] = None

@dataclass
class BreakStatement(Statement):
    pass

@dataclass
class ContinueStatement(Statement):
    pass

@dataclass
class Block(Statement):
    statements: List[Statement]

@dataclass
class ExpressionStatement(Statement):
    expression: 'Expression'

@dataclass
class Expression(ASTNode):
    pass

@dataclass
class BinaryExpression(Expression):
    left: Expression
    operator: str
    right: Expression

@dataclass
class UnaryExpression(Expression):
    operator: str
    operand: Expression
    is_postfix: bool = False

@dataclass
class FunctionCall(Expression):
    name: str
    arguments: List[Expression]

@dataclass
class ArrayAccess(Expression):
    array: Expression
    index: Expression

@dataclass
class Identifier(Expression):
    name: str

@dataclass
class Literal(Expression):
    value: Any
    type: str

# =============================================================================
# ENHANCED PARSER (RECURSIVE DESCENT)
# =============================================================================

class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"Parse Error at line {token.line}, column {token.column}: {message}")

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        self.errors = []
    
    def current_token(self) -> Token:
        if self.current >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[self.current]
    
    def peek_token(self, offset: int = 1) -> Token:
        pos = self.current + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[pos]
    
    def advance(self) -> Token:
        token = self.current_token()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token
    
    def match(self, *token_types: TokenType) -> bool:
        return self.current_token().type in token_types
    
    def consume(self, token_type: TokenType, message: str) -> Token:
        if self.current_token().type == token_type:
            return self.advance()
        
        error = ParseError(message, self.current_token())
        self.errors.append(error)
        return self.current_token()
    
    def synchronize(self):
        """Error recovery mechanism"""
        self.advance()
        while not self.match(TokenType.EOF):
            if self.tokens[self.current - 1].type == TokenType.SEMICOLON:
                return
            if self.match(TokenType.INT, TokenType.FLOAT_KW, TokenType.BOOL, TokenType.STRING_KW,
                          TokenType.VOID, TokenType.IF, TokenType.WHILE, TokenType.FOR, 
                          TokenType.RETURN, TokenType.BREAK, TokenType.CONTINUE):
                return
            self.advance()
    
    def parse(self) -> Program:
        functions = []
        while not self.match(TokenType.EOF):
            try:
                func = self.parse_function()
                if func:
                    functions.append(func)
            except ParseError as e:
                self.errors.append(e)
                self.synchronize()
        
        return Program(functions)
    
    def parse_function(self) -> Optional[Function]:
        # Parse return type
        if not self.match(TokenType.INT, TokenType.FLOAT_KW, TokenType.BOOL, 
                          TokenType.STRING_KW, TokenType.VOID):
            return None
        
        return_type = self.advance().value
        
        # Parse function name
        name_token = self.consume(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.value
        
        # Parse parameters
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after function name")
        parameters = self.parse_parameter_list()
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters")
        
        # Parse body
        body = self.parse_block()
        
        return Function(name, parameters, return_type, body.statements)
    
    def parse_parameter_list(self) -> List[Parameter]:
        parameters = []
        
        if not self.match(TokenType.RIGHT_PAREN):
            parameters.append(self.parse_parameter())
            
            while self.match(TokenType.COMMA):
                self.advance()
                parameters.append(self.parse_parameter())
        
        return parameters
    
    def parse_parameter(self) -> Parameter:
        # Parse parameter type
        if not self.match(TokenType.INT, TokenType.FLOAT_KW, TokenType.BOOL, TokenType.STRING_KW):
            self.consume(TokenType.INT, "Expected parameter type")
            param_type = "int"  # Default fallback
        else:
            param_type = self.advance().value
        
        param_name = self.consume(TokenType.IDENTIFIER, "Expected parameter name").value
        
        # Check for array parameter
        is_array = False
        array_size = None
        if self.match(TokenType.LEFT_BRACKET):
            is_array = True
            self.advance()
            if self.match(TokenType.INTEGER):
                array_size = int(self.advance().value)
            self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after array size")
        
        return Parameter(param_name, param_type, is_array, array_size)
    
    def parse_block(self) -> Block:
        self.consume(TokenType.LEFT_BRACE, "Expected '{'")
        statements = []
        
        while not self.match(TokenType.RIGHT_BRACE, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        self.consume(TokenType.RIGHT_BRACE, "Expected '}'")
        return Block(statements)
    
    def parse_statement(self) -> Optional[Statement]:
        try:
            # Variable declaration
            if self.match(TokenType.INT, TokenType.FLOAT_KW, TokenType.BOOL, TokenType.STRING_KW):
                return self.parse_var_declaration()
            
            # Control flow statements
            if self.match(TokenType.IF):
                return self.parse_if_statement()
            
            if self.match(TokenType.WHILE):
                return self.parse_while_statement()
            
            if self.match(TokenType.FOR):
                return self.parse_for_statement()
            
            if self.match(TokenType.RETURN):
                return self.parse_return_statement()
            
            if self.match(TokenType.BREAK):
                self.advance()
                self.consume(TokenType.SEMICOLON, "Expected ';' after break")
                return BreakStatement()
            
            if self.match(TokenType.CONTINUE):
                self.advance()
                self.consume(TokenType.SEMICOLON, "Expected ';' after continue")
                return ContinueStatement()
            
            # Block
            if self.match(TokenType.LEFT_BRACE):
                return self.parse_block()
            
            # Assignment or expression statement
            return self.parse_assignment_or_expression()
        
        except ParseError as e:
            self.errors.append(e)
            self.synchronize()
            return None
    
    def parse_var_declaration(self) -> VarDeclaration:
        var_type = self.advance().value
        var_name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        
        # Check for array declaration
        is_array = False
        array_size = None
        if self.match(TokenType.LEFT_BRACKET):
            is_array = True
            self.advance()
            if self.match(TokenType.INTEGER):
                array_size = int(self.advance().value)
            self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after array size")
        
        initializer = None
        if self.match(TokenType.ASSIGN):
            self.advance()
            initializer = self.parse_expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
        return VarDeclaration(var_name, var_type, is_array, array_size, initializer)
    
    def parse_assignment_or_expression(self) -> Statement:
        # Parse the left-hand side first
        expr = self.parse_expression()
        
        # Check if it's an assignment
        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                      TokenType.MULTIPLY_ASSIGN, TokenType.DIVIDE_ASSIGN):
            operator = self.advance().value
            value = self.parse_expression()
            
            # Convert compound assignment to simple assignment
            if operator != "=":
                op_map = {"+=": "+", "-=": "-", "*=": "*", "/=": "/"}
                value = BinaryExpression(expr, op_map[operator], value)
            
            self.consume(TokenType.SEMICOLON, "Expected ';' after assignment")
            return Assignment(expr, value)
        else:
            self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
            return ExpressionStatement(expr)
    
    def parse_if_statement(self) -> IfStatement:
        self.advance()  # consume 'if'
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'if'")
        condition = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after if condition")
        
        then_stmt = self.parse_statement()
        else_stmt = None
        
        if self.match(TokenType.ELSE):
            self.advance()
            else_stmt = self.parse_statement()
        
        return IfStatement(condition, then_stmt, else_stmt)
    
    def parse_while_statement(self) -> WhileStatement:
        self.advance()  # consume 'while'
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'while'")
        condition = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after while condition")
        body = self.parse_statement()
        return WhileStatement(condition, body)
    
    def parse_for_statement(self) -> ForStatement:
        self.advance()  # consume 'for'
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'for'")
        
        # Parse init (can be declaration or expression)
        init = None
        if not self.match(TokenType.SEMICOLON):
            if self.match(TokenType.INT, TokenType.FLOAT_KW, TokenType.BOOL, TokenType.STRING_KW):
                init = self.parse_var_declaration()
            else:
                init = ExpressionStatement(self.parse_expression())
                self.consume(TokenType.SEMICOLON, "Expected ';' after for init")
        else:
            self.advance()  # consume ';'
        
        # Parse condition
        condition = None
        if not self.match(TokenType.SEMICOLON):
            condition = self.parse_expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after for condition")
        
        # Parse update
        update = None
        if not self.match(TokenType.RIGHT_PAREN):
            update = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after for clauses")
        
        body = self.parse_statement()
        return ForStatement(init, condition, update, body)
    
    def parse_return_statement(self) -> ReturnStatement:
        self.advance()  # consume 'return'
        value = None
        
        if not self.match(TokenType.SEMICOLON):
            value = self.parse_expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after return statement")
        return ReturnStatement(value)
    
    def parse_expression(self) -> Expression:
        return self.parse_logical_or()
    
    def parse_logical_or(self) -> Expression:
        expr = self.parse_logical_and()
        
        while self.match(TokenType.LOGICAL_OR):
            operator = self.advance().value
            right = self.parse_logical_and()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_logical_and(self) -> Expression:
        expr = self.parse_equality()
        
        while self.match(TokenType.LOGICAL_AND):
            operator = self.advance().value
            right = self.parse_equality()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_equality(self) -> Expression:
        expr = self.parse_comparison()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.advance().value
            right = self.parse_comparison()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_comparison(self) -> Expression:
        expr = self.parse_term()
        
        while self.match(TokenType.GREATER_THAN, TokenType.GREATER_EQUAL,
                         TokenType.LESS_THAN, TokenType.LESS_EQUAL):
            operator = self.advance().value
            right = self.parse_term()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_term(self) -> Expression:
        expr = self.parse_factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.advance().value
            right = self.parse_factor()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_factor(self) -> Expression:
        expr = self.parse_unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            operator = self.advance().value
            right = self.parse_unary()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_unary(self) -> Expression:
        if self.match(TokenType.LOGICAL_NOT, TokenType.MINUS, TokenType.INCREMENT, TokenType.DECREMENT):
            operator = self.advance().value
            expr = self.parse_unary()
            return UnaryExpression(operator, expr, False)
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> Expression:
        expr = self.parse_primary()
        
        while True:
            if self.match(TokenType.LEFT_BRACKET):
                # Array access
                self.advance()
                index = self.parse_expression()
                self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after array index")
                expr = ArrayAccess(expr, index)
            elif self.match(TokenType.INCREMENT, TokenType.DECREMENT):
                # Postfix increment/decrement
                operator = self.advance().value
                expr = UnaryExpression(operator, expr, True)
            else:
                break
        
        return expr
    
    def parse_primary(self) -> Expression:
        # Literals
        if self.match(TokenType.INTEGER):
            token = self.advance()
            return Literal(int(token.value), "int")
        
        if self.match(TokenType.FLOAT):
            token = self.advance()
            return Literal(float(token.value), "float")
        
        if self.match(TokenType.BOOLEAN):
            token = self.advance()
            return Literal(token.value == "true", "bool")
        
        if self.match(TokenType.STRING):
            token = self.advance()
            return Literal(token.value, "string")
        
        # Identifiers and function calls
        if self.match(TokenType.IDENTIFIER):
            name = self.advance().value
            
            # Function call
            if self.match(TokenType.LEFT_PAREN):
                self.advance()
                arguments = []
                
                if not self.match(TokenType.RIGHT_PAREN):
                    arguments.append(self.parse_expression())
                    
                    while self.match(TokenType.COMMA):
                        self.advance()
                        arguments.append(self.parse_expression())
                
                self.consume(TokenType.RIGHT_PAREN, "Expected ')' after function arguments")
                return FunctionCall(name, arguments)
            
            # Variable reference
            return Identifier(name)
        
        # Parenthesized expression
        if self.match(TokenType.LEFT_PAREN):
            self.advance()
            expr = self.parse_expression()
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after expression")
            return expr
        
        raise ParseError("Expected expression", self.current_token())

# =============================================================================
# ENHANCED SYMBOL TABLE AND SEMANTIC ANALYZER
# =============================================================================

@dataclass
class Symbol:
    name: str
    type: str
    kind: str  # 'variable', 'function', 'parameter', 'array'
    scope_level: int
    initialized: bool = False
    line: int = 0
    column: int = 0
    is_array: bool = False
    array_size: Optional[int] = None
    used: bool = False

class SymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]  # Global scope
        self.scope_level = 0
        self.all_symbols = []
    
    def enter_scope(self):
        self.scope_level += 1
        self.scopes.append({})
    
    def exit_scope(self):
        if self.scope_level > 0:
            self.scope_level -= 1
    
    def declare(self, symbol: Symbol) -> bool:
        """Returns True if declaration successful, False if already declared in current scope"""
        current_scope = self.scopes[self.scope_level]
        if symbol.name in current_scope:
            return False
        
        symbol.scope_level = self.scope_level
        current_scope[symbol.name] = symbol
        self.all_symbols.append(symbol)
        return True
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up symbol in all scopes from current to global"""
        for i in range(len(self.scopes) - 1, -1, -1):
            if i <= self.scope_level and name in self.scopes[i]:
                symbol = self.scopes[i][name]
                symbol.used = True
                return symbol
        return None
    
    def lookup_current_scope(self, name: str) -> Optional[Symbol]:
        """Look up symbol only in current scope"""
        if self.scope_level < len(self.scopes):
            return self.scopes[self.scope_level].get(name)
        return None

class SemanticError(Exception):
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Semantic Error at line {line}, column {column}: {message}")

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors = []
        self.current_function: Optional[Function] = None
        self.current_return_type: Optional[str] = None
        self.loop_depth = 0
        self.type_compatibility = {
            ('int', 'float'): 'float',
            ('float', 'int'): 'float',
            ('int', 'int'): 'int',
            ('float', 'float'): 'float',
            ('bool', 'bool'): 'bool',
            ('string', 'string'): 'string'
        }
    
    def error(self, message: str, line: int = 0, column: int = 0):
        error = SemanticError(message, line, column)
        self.errors.append(error)
    
    def analyze(self, program: Program) -> bool:
        """Returns True if no semantic errors found"""
        self.visit_program(program)
        return len(self.errors) == 0
    
    def visit_program(self, node: Program):
        # First pass: declare all functions in global scope
        for function in node.functions:
            self.declare_function(function)
        
        # Check for main function
        main_symbol = self.symbol_table.lookup('main')
        if not main_symbol or main_symbol.kind != 'function':
            self.error("Program must have a main function")
        
        # Second pass: analyze function bodies
        for function in node.functions:
            self.visit_function(function)
    
    def declare_function(self, node: Function):
        symbol = Symbol(
            name=node.name,
            type=node.return_type,
            kind='function',
            scope_level=0,
            initialized=True
        )
        
        if not self.symbol_table.declare(symbol):
            self.error(f"Function '{node.name}' already declared")
    
    def visit_function(self, node: Function):
        self.current_function = node
        self.current_return_type = node.return_type
        
        # Enter function scope
        self.symbol_table.enter_scope()
        
        # Declare parameters in function scope
        for param in node.parameters:
            symbol = Symbol(
                name=param.name,
                type=param.type,
                kind='parameter',
                scope_level=self.symbol_table.scope_level,
                initialized=True,
                is_array=param.is_array,
                array_size=param.array_size
            )
            
            if not self.symbol_table.declare(symbol):
                self.error(f"Parameter '{param.name}' already declared in function '{node.name}'")
        
        # Analyze function body
        for stmt in node.body:
            self.visit_statement(stmt)
        
        # Check if non-void function has return statement
        if node.return_type != 'void' and not self.has_return_statement(node.body):
            self.error(f"Function '{node.name}' must return a value of type '{node.return_type}'")
        
        # Exit function scope
        self.symbol_table.exit_scope()
        self.current_function = None
        self.current_return_type = None
    
    def visit_statement(self, node: Statement):
        if isinstance(node, VarDeclaration):
            self.visit_var_declaration(node)
        elif isinstance(node, Assignment):
            self.visit_assignment(node)
        elif isinstance(node, IfStatement):
            self.visit_if_statement(node)
        elif isinstance(node, WhileStatement):
            self.visit_while_statement(node)
        elif isinstance(node, ForStatement):
            self.visit_for_statement(node)
        elif isinstance(node, ReturnStatement):
            self.visit_return_statement(node)
        elif isinstance(node, BreakStatement):
            self.visit_break_statement(node)
        elif isinstance(node, ContinueStatement):
            self.visit_continue_statement(node)
        elif isinstance(node, Block):
            self.visit_block(node)
        elif isinstance(node, ExpressionStatement):
            self.visit_expression(node.expression)
    
    def visit_var_declaration(self, node: VarDeclaration):
        # Check if variable already declared in current scope
        if self.symbol_table.lookup_current_scope(node.name):
            self.error(f"Variable '{node.name}' already declared in current scope")
            return
        
        # Validate array size
        if node.is_array and node.array_size is not None and node.array_size <= 0:
            self.error(f"Array size must be positive, got {node.array_size}")
            return
        
        symbol = Symbol(
            name=node.name,
            type=node.type,
            kind='array' if node.is_array else 'variable',
            scope_level=self.symbol_table.scope_level,
            initialized=node.initializer is not None,
            is_array=node.is_array,
            array_size=node.array_size
        )
        
        if not self.symbol_table.declare(symbol):
            self.error(f"Variable '{node.name}' already declared in current scope")
            return
        
        # Type check initializer
        if node.initializer:
            init_type = self.visit_expression(node.initializer)
            if node.is_array:
                self.error(f"Array initialization not yet supported")
            elif not self.is_compatible_type(node.type, init_type):
                self.error(f"Cannot assign {init_type} to {node.type} variable '{node.name}'")
    
    def visit_assignment(self, node: Assignment):
        # Check target (left-hand side)
        target_type = None
        if isinstance(node.target, Identifier):
            symbol = self.symbol_table.lookup(node.target.name)
            if not symbol:
                self.error(f"Undefined variable '{node.target.name}'")
                return
            
            if symbol.kind == 'function':
                self.error(f"Cannot assign to function '{node.target.name}'")
                return
            
            target_type = symbol.type
            symbol.initialized = True
            
        elif isinstance(node.target, ArrayAccess):
            array_type = self.visit_expression(node.target.array)
            index_type = self.visit_expression(node.target.index)
            
            if index_type != 'int':
                self.error(f"Array index must be integer, got {index_type}")
            
            # Remove array suffix to get element type
            target_type = array_type.replace('[]', '') if '[]' in array_type else array_type
        else:
            self.error("Invalid assignment target")
            return
        
        # Type check assignment
        value_type = self.visit_expression(node.value)
        if target_type and not self.is_compatible_type(target_type, value_type):
            self.error(f"Cannot assign {value_type} to {target_type}")
    
    def visit_if_statement(self, node: IfStatement):
        condition_type = self.visit_expression(node.condition)
        if condition_type != 'bool':
            self.error(f"If condition must be boolean, got {condition_type}")
        
        if isinstance(node.then_stmt, Block):
            self.symbol_table.enter_scope()
            self.visit_statement(node.then_stmt)
            self.symbol_table.exit_scope()
        else:
            self.visit_statement(node.then_stmt)
        
        if node.else_stmt:
            if isinstance(node.else_stmt, Block):
                self.symbol_table.enter_scope()
                self.visit_statement(node.else_stmt)
                self.symbol_table.exit_scope()
            else:
                self.visit_statement(node.else_stmt)
    
    def visit_while_statement(self, node: WhileStatement):
        condition_type = self.visit_expression(node.condition)
        if condition_type != 'bool':
            self.error(f"While condition must be boolean, got {condition_type}")
        
        self.loop_depth += 1
        if isinstance(node.body, Block):
            self.symbol_table.enter_scope()
            self.visit_statement(node.body)
            self.symbol_table.exit_scope()
        else:
            self.visit_statement(node.body)
        self.loop_depth -= 1
    
    def visit_for_statement(self, node: ForStatement):
        self.symbol_table.enter_scope()
        
        if node.init:
            self.visit_statement(node.init)
        
        if node.condition:
            condition_type = self.visit_expression(node.condition)
            if condition_type != 'bool':
                self.error(f"For condition must be boolean, got {condition_type}")
        
        if node.update:
            self.visit_expression(node.update)
        
        self.loop_depth += 1
        self.visit_statement(node.body)
        self.loop_depth -= 1
        
        self.symbol_table.exit_scope()
    
    def visit_return_statement(self, node: ReturnStatement):
        if not self.current_function:
            self.error("Return statement outside function")
            return
        
        expected_type = self.current_return_type
        
        if node.value:
            actual_type = self.visit_expression(node.value)
            if not self.is_compatible_type(expected_type, actual_type):
                self.error(f"Return type mismatch: expected {expected_type}, got {actual_type}")
        else:
            if expected_type != 'void':
                self.error(f"Function '{self.current_function.name}' must return a value")
    
    def visit_break_statement(self, node: BreakStatement):
        if self.loop_depth == 0:
            self.error("Break statement outside loop")
    
    def visit_continue_statement(self, node: ContinueStatement):
        if self.loop_depth == 0:
            self.error("Continue statement outside loop")
    
    def visit_block(self, node: Block):
        self.symbol_table.enter_scope()
        for stmt in node.statements:
            self.visit_statement(stmt)
        self.symbol_table.exit_scope()
    
    def visit_expression(self, node: Expression) -> str:
        """Returns the type of the expression"""
        if isinstance(node, Literal):
            return node.type
        elif isinstance(node, Identifier):
            return self.visit_identifier(node)
        elif isinstance(node, BinaryExpression):
            return self.visit_binary_expression(node)
        elif isinstance(node, UnaryExpression):
            return self.visit_unary_expression(node)
        elif isinstance(node, FunctionCall):
            return self.visit_function_call(node)
        elif isinstance(node, ArrayAccess):
            return self.visit_array_access(node)
        else:
            self.error(f"Unknown expression type: {type(node)}")
            return "error"
    
    def visit_identifier(self, node: Identifier) -> str:
        symbol = self.symbol_table.lookup(node.name)
        if not symbol:
            self.error(f"Undefined identifier '{node.name}'")
            return "error"
        
        if symbol.kind == 'variable' and not symbol.initialized:
            self.error(f"Variable '{node.name}' used before initialization")
        
        return symbol.type + ('[]' if symbol.is_array else '')
    
    def visit_array_access(self, node: ArrayAccess) -> str:
        array_type = self.visit_expression(node.array)
        index_type = self.visit_expression(node.index)
        
        if index_type != 'int':
            self.error(f"Array index must be integer, got {index_type}")
        
        if not array_type.endswith('[]'):
            self.error(f"Subscript operator applied to non-array type {array_type}")
            return "error"
        
        # Return element type (remove [] suffix)
        return array_type[:-2]
    
    def visit_binary_expression(self, node: BinaryExpression) -> str:
        left_type = self.visit_expression(node.left)
        right_type = self.visit_expression(node.right)
        
        # Arithmetic operators
        if node.operator in ['+', '-', '*', '/', '%']:
            if left_type in ['int', 'float'] and right_type in ['int', 'float']:
                return self.type_compatibility.get((left_type, right_type), 'error')
            else:
                self.error(f"Arithmetic operator '{node.operator}' requires numeric operands, got {left_type} and {right_type}")
                return "error"
        
        # Comparison operators
        elif node.operator in ['<', '>', '<=', '>=']:
            if left_type in ['int', 'float'] and right_type in ['int', 'float']:
                return 'bool'
            else:
                self.error(f"Comparison operator '{node.operator}' requires numeric operands, got {left_type} and {right_type}")
                return "error"
        
        # Equality operators
        elif node.operator in ['==', '!=']:
            if self.is_compatible_type(left_type, right_type) or self.is_compatible_type(right_type, left_type):
                return 'bool'
            else:
                self.error(f"Equality operator '{node.operator}' requires compatible types, got {left_type} and {right_type}")
                return "error"
        
        # Logical operators
        elif node.operator in ['&&', '||']:
            if left_type == 'bool' and right_type == 'bool':
                return 'bool'
            else:
                self.error(f"Logical operator '{node.operator}' requires boolean operands, got {left_type} and {right_type}")
                return "error"
        
        else:
            self.error(f"Unknown binary operator: {node.operator}")
            return "error"
    
    def visit_unary_expression(self, node: UnaryExpression) -> str:
        operand_type = self.visit_expression(node.operand)
        
        if node.operator in ['-']:
            if operand_type in ['int', 'float']:
                return operand_type
            else:
                self.error(f"Unary minus requires numeric operand, got {operand_type}")
                return "error"
        
        elif node.operator == '!':
            if operand_type == 'bool':
                return 'bool'
            else:
                self.error(f"Logical not requires boolean operand, got {operand_type}")
                return "error"
        
        elif node.operator in ['++', '--']:
            if operand_type in ['int', 'float']:
                # Check if operand is assignable
                if isinstance(node.operand, Identifier):
                    symbol = self.symbol_table.lookup(node.operand.name)
                    if symbol and symbol.kind == 'variable':
                        return operand_type
                self.error(f"Increment/decrement requires assignable operand")
                return "error"
            else:
                self.error(f"Increment/decrement requires numeric operand, got {operand_type}")
                return "error"
        
        else:
            self.error(f"Unknown unary operator: {node.operator}")
            return "error"
    
    def visit_function_call(self, node: FunctionCall) -> str:
        symbol = self.symbol_table.lookup(node.name)
        if not symbol:
            self.error(f"Undefined function '{node.name}'")
            return "error"
        
        if symbol.kind != 'function':
            self.error(f"'{node.name}' is not a function")
            return "error"
        
        # Find the function definition to check parameters
        function_def = None
        for func in self.get_all_functions():
            if func.name == node.name:
                function_def = func
                break
        
        if not function_def:
            self.error(f"Function '{node.name}' definition not found")
            return symbol.type
        
        # Check number of arguments
        if len(node.arguments) != len(function_def.parameters):
            self.error(f"Function '{node.name}' expects {len(function_def.parameters)} arguments, got {len(node.arguments)}")
            return symbol.type
        
        # Check argument types
        for i, (arg, param) in enumerate(zip(node.arguments, function_def.parameters)):
            arg_type = self.visit_expression(arg)
            expected_type = param.type + ('[]' if param.is_array else '')
            if not self.is_compatible_type(expected_type, arg_type):
                self.error(f"Argument {i+1} to function '{node.name}': expected {expected_type}, got {arg_type}")
        
        return symbol.type
    
    def is_compatible_type(self, target_type: str, source_type: str) -> bool:
        """Check if source_type can be assigned to target_type"""
        if target_type == source_type:
            return True
        
        # Allow int to float conversion
        if target_type == 'float' and source_type == 'int':
            return True
        
        return False
    
    def get_all_functions(self) -> List[Function]:
        """Helper method to get all function definitions"""
        return getattr(self, '_functions', [])
    
    def set_functions(self, functions: List[Function]):
        """Set the list of functions for reference during semantic analysis"""
        self._functions = functions
    
    def has_return_statement(self, statements: List[Statement]) -> bool:
        """Check if statements contain a return statement"""
        for stmt in statements:
            if isinstance(stmt, ReturnStatement):
                return True
            elif isinstance(stmt, Block):
                if self.has_return_statement(stmt.statements):
                    return True
            elif isinstance(stmt, IfStatement):
                if (stmt.else_stmt and 
                    self.has_return_statement([stmt.then_stmt]) and 
                    self.has_return_statement([stmt.else_stmt])):
                    return True
        return False
    
    def check_unused_variables(self):
        """Check for unused variables and issue warnings"""
        pass
    
    def get_unused_variables(self) -> List[str]:
        """Get list of unused variables as a separate method""" 
        unused = []
        for symbol in self.symbol_table.all_symbols:
            if symbol.kind in ['variable'] and not symbol.used:
                unused.append(symbol.name)
        return unused
# =============================================================================
# ENHANCED INTERMEDIATE CODE GENERATOR (THREE ADDRESS CODE)
# =============================================================================

@dataclass
class ThreeAddressCode:
    operation: str
    arg1: Optional[str]
    arg2: Optional[str]
    result: str
    line_number: int = 0

@dataclass
class BasicBlock:
    label: str
    instructions: List[ThreeAddressCode] = field(default_factory=list)
    successors: List['BasicBlock'] = field(default_factory=list)
    predecessors: List['BasicBlock'] = field(default_factory=list)

class IntermediateCodeGenerator:
    def __init__(self):
        self.code: List[ThreeAddressCode] = []
        self.temp_counter = 0
        self.label_counter = 0
        self.line_number = 0
        self.break_labels: List[str] = []
        self.continue_labels: List[str] = []
    
    def new_temp(self) -> str:
        self.temp_counter += 1
        return f"t{self.temp_counter}"
    
    def new_label(self) -> str:
        self.label_counter += 1
        return f"L{self.label_counter}"
    
    def emit(self, op: str, arg1: str = None, arg2: str = None, result: str = None):
        self.line_number += 1
        self.code.append(ThreeAddressCode(op, arg1, arg2, result, self.line_number))
    
    def generate(self, program: Program) -> List[ThreeAddressCode]:
        for function in program.functions:
            self.generate_function(function)
        return self.code
    
    def generate_function(self, node: Function):
        self.emit("FUNCTION", None, None, node.name)
        
        # Generate parameter declarations
        for param in node.parameters:
            if param.is_array:
                self.emit("PARAM_ARRAY", param.name, str(param.array_size or 0), param.type)
            else:
                self.emit("PARAM", param.name, None, param.type)
        
        for stmt in node.body:
            self.generate_statement(stmt)
        
        self.emit("END_FUNCTION", None, None, node.name)
    
    def generate_statement(self, node: Statement):
        if isinstance(node, VarDeclaration):
            if node.is_array:
                self.emit("DECLARE_ARRAY", node.name, str(node.array_size or 0), node.type)
            else:
                self.emit("DECLARE", node.name, None, node.type)
            
            if node.initializer:
                temp = self.generate_expression(node.initializer)
                self.emit("ASSIGN", temp, None, node.name)
        
        elif isinstance(node, Assignment):
            if isinstance(node.target, Identifier):
                temp = self.generate_expression(node.value)
                self.emit("ASSIGN", temp, None, node.target.name)
            elif isinstance(node.target, ArrayAccess):
                array_temp = self.generate_expression(node.target.array)
                index_temp = self.generate_expression(node.target.index)
                value_temp = self.generate_expression(node.value)
                self.emit("ARRAY_ASSIGN", array_temp, index_temp, value_temp)
        
        elif isinstance(node, IfStatement):
            condition_temp = self.generate_expression(node.condition)
            else_label = self.new_label()
            end_label = self.new_label()
            
            self.emit("IF_FALSE", condition_temp, None, else_label)
            self.generate_statement(node.then_stmt)
            self.emit("GOTO", None, None, end_label)
            self.emit("LABEL", None, None, else_label)
            
            if node.else_stmt:
                self.generate_statement(node.else_stmt)
            
            self.emit("LABEL", None, None, end_label)
        
        elif isinstance(node, WhileStatement):
            start_label = self.new_label()
            end_label = self.new_label()
            
            self.break_labels.append(end_label)
            self.continue_labels.append(start_label)
            
            self.emit("LABEL", None, None, start_label)
            condition_temp = self.generate_expression(node.condition)
            self.emit("IF_FALSE", condition_temp, None, end_label)
            self.generate_statement(node.body)
            self.emit("GOTO", None, None, start_label)
            self.emit("LABEL", None, None, end_label)
            
            self.break_labels.pop()
            self.continue_labels.pop()
        
        elif isinstance(node, ForStatement):
            start_label = self.new_label()
            condition_label = self.new_label()
            update_label = self.new_label()
            end_label = self.new_label()
            
            self.break_labels.append(end_label)
            self.continue_labels.append(update_label)
            
            if node.init:
                self.generate_statement(node.init)
            
            self.emit("GOTO", None, None, condition_label)
            self.emit("LABEL", None, None, start_label)
            self.generate_statement(node.body)
            
            self.emit("LABEL", None, None, update_label)
            if node.update:
                self.generate_expression(node.update)
            
            self.emit("LABEL", None, None, condition_label)
            if node.condition:
                condition_temp = self.generate_expression(node.condition)
                self.emit("IF_TRUE", condition_temp, None, start_label)
            else:
                self.emit("GOTO", None, None, start_label)
            
            self.emit("LABEL", None, None, end_label)
            
            self.break_labels.pop()
            self.continue_labels.pop()
        
        elif isinstance(node, ReturnStatement):
            if node.value:
                temp = self.generate_expression(node.value)
                self.emit("RETURN", temp, None, None)
            else:
                self.emit("RETURN", None, None, None)
        
        elif isinstance(node, BreakStatement):
            if self.break_labels:
                self.emit("GOTO", None, None, self.break_labels[-1])
        
        elif isinstance(node, ContinueStatement):
            if self.continue_labels:
                self.emit("GOTO", None, None, self.continue_labels[-1])
        
        elif isinstance(node, Block):
            for stmt in node.statements:
                self.generate_statement(stmt)
        
        elif isinstance(node, ExpressionStatement):
            self.generate_expression(node.expression)
    
    def generate_expression(self, node: Expression) -> str:
        if isinstance(node, Literal):
            return str(node.value)
        
        elif isinstance(node, Identifier):
            return node.name
        
        elif isinstance(node, ArrayAccess):
            array_temp = self.generate_expression(node.array)
            index_temp = self.generate_expression(node.index)
            result_temp = self.new_temp()
            self.emit("ARRAY_ACCESS", array_temp, index_temp, result_temp)
            return result_temp
        
        elif isinstance(node, BinaryExpression):
            left_temp = self.generate_expression(node.left)
            right_temp = self.generate_expression(node.right)
            result_temp = self.new_temp()
            self.emit(node.operator, left_temp, right_temp, result_temp)
            return result_temp
        
        elif isinstance(node, UnaryExpression):
            operand_temp = self.generate_expression(node.operand)
            result_temp = self.new_temp()
            
            if node.is_postfix:
                # For postfix, return current value then modify
                self.emit("ASSIGN", operand_temp, None, result_temp)
                if node.operator == '++':
                    self.emit("+", operand_temp, "1", operand_temp)
                elif node.operator == '--':
                    self.emit("-", operand_temp, "1", operand_temp)
            else:
                # For prefix, modify then return
                if node.operator == '++':
                    self.emit("+", operand_temp, "1", operand_temp)
                    self.emit("ASSIGN", operand_temp, None, result_temp)
                elif node.operator == '--':
                    self.emit("-", operand_temp, "1", operand_temp)
                    self.emit("ASSIGN", operand_temp, None, result_temp)
                else:
                    self.emit(node.operator, operand_temp, None, result_temp)
            
            return result_temp
        
        elif isinstance(node, FunctionCall):
            # Generate code for arguments
            for arg in node.arguments:
                arg_temp = self.generate_expression(arg)
                self.emit("PARAM", arg_temp, None, None)
            
            result_temp = self.new_temp()
            self.emit("CALL", node.name, str(len(node.arguments)), result_temp)
            return result_temp
        
        return "unknown"

# =============================================================================
# CODE OPTIMIZER
# =============================================================================

class CodeOptimizer:
    def __init__(self):
        self.optimizations_applied = []
    
    def optimize(self, code: List[ThreeAddressCode]) -> List[ThreeAddressCode]:
        """Apply various optimization techniques"""
        optimized_code = code.copy()
        
        # Apply optimizations in sequence
        optimized_code = self.constant_folding(optimized_code)
        optimized_code = self.copy_propagation(optimized_code)
        optimized_code = self.dead_code_elimination(optimized_code)
        optimized_code = self.algebraic_simplification(optimized_code)
        optimized_code = self.remove_redundant_jumps(optimized_code)
        
        return optimized_code
    
    def constant_folding(self, code: List[ThreeAddressCode]) -> List[ThreeAddressCode]:
        """Fold constant expressions"""
        optimized = []
        
        for instr in code:
            if instr.operation in ['+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=']:
                if (instr.arg1 and instr.arg2 and 
                    self.is_constant(instr.arg1) and self.is_constant(instr.arg2)):
                    
                    val1 = self.get_constant_value(instr.arg1)
                    val2 = self.get_constant_value(instr.arg2)
                    
                    try:
                        if instr.operation == '+':
                            result = val1 + val2
                        elif instr.operation == '-':
                            result = val1 - val2
                        elif instr.operation == '*':
                            result = val1 * val2
                        elif instr.operation == '/':
                            if val2 != 0:
                                result = val1 / val2
                            else:
                                optimized.append(instr)
                                continue
                        elif instr.operation == '%':
                            if val2 != 0:
                                result = val1 % val2
                            else:
                                optimized.append(instr)
                                continue
                        elif instr.operation == '==':
                            result = val1 == val2
                        elif instr.operation == '!=':
                            result = val1 != val2
                        elif instr.operation == '<':
                            result = val1 < val2
                        elif instr.operation == '>':
                            result = val1 > val2
                        elif instr.operation == '<=':
                            result = val1 <= val2
                        elif instr.operation == '>=':
                            result = val1 >= val2
                        else:
                            optimized.append(instr)
                            continue
                        
                        # Replace with constant assignment
                        new_instr = ThreeAddressCode("ASSIGN", str(result), None, instr.result, instr.line_number)
                        optimized.append(new_instr)
                        self.optimizations_applied.append(f"Constant folding: {instr.arg1} {instr.operation} {instr.arg2} = {result}")
                        continue
                    except:
                        pass
            
            optimized.append(instr)
        
        return optimized
    
    def copy_propagation(self, code: List[ThreeAddressCode]) -> List[ThreeAddressCode]:
        """Propagate copies (x = y; use x -> use y)"""
        optimized = []
        copies = {}  # var -> value
        
        for instr in code:
            # Update copies based on current instruction
            if instr.operation == "ASSIGN" and instr.arg2 is None:
                if self.is_variable(instr.arg1):
                    copies[instr.result] = instr.arg1
                else:
                    copies[instr.result] = instr.arg1
            elif instr.result and instr.result in copies:
                # Variable is redefined, remove from copies
                del copies[instr.result]
            
            # Apply copy propagation
            new_instr = copy.deepcopy(instr)
            if new_instr.arg1 and new_instr.arg1 in copies:
                new_instr.arg1 = copies[new_instr.arg1]
                self.optimizations_applied.append(f"Copy propagation: replaced {instr.arg1} with {copies[instr.arg1]}")
            
            if new_instr.arg2 and new_instr.arg2 in copies:
                new_instr.arg2 = copies[new_instr.arg2]
                self.optimizations_applied.append(f"Copy propagation: replaced {instr.arg2} with {copies[instr.arg2]}")
            
            optimized.append(new_instr)
        
        return optimized
    
    def dead_code_elimination(self, code: List[ThreeAddressCode]) -> List[ThreeAddressCode]:
        """Remove dead code (unused assignments)"""
        # Build use-def chains
        used_vars = set()
        defined_vars = set()
        
        # First pass: collect all used and defined variables
        for instr in code:
            if instr.arg1 and self.is_variable(instr.arg1):
                used_vars.add(instr.arg1)
            if instr.arg2 and self.is_variable(instr.arg2):
                used_vars.add(instr.arg2)
            if instr.result and self.is_variable(instr.result):
                defined_vars.add(instr.result)
        
        # Second pass: remove unused definitions
        optimized = []
        for instr in code:
            # Keep instruction if:
            # 1. It's not an assignment, or
            # 2. The result is used, or
            # 3. It has side effects
            if (instr.operation != "ASSIGN" or 
                not instr.result or 
                instr.result in used_vars or
                instr.operation in ["CALL", "RETURN", "PARAM", "LABEL", "GOTO", "IF_TRUE", "IF_FALSE"]):
                optimized.append(instr)
            else:
                self.optimizations_applied.append(f"Dead code elimination: removed {instr.operation} {instr.result}")
        
        return optimized
    
    def algebraic_simplification(self, code: List[ThreeAddressCode]) -> List[ThreeAddressCode]:
        """Apply algebraic simplifications"""
        optimized = []
        
        for instr in code:
            simplified = False
            
            if instr.operation == '+':
                # x + 0 = x, 0 + x = x
                if instr.arg2 == '0':
                    new_instr = ThreeAddressCode("ASSIGN", instr.arg1, None, instr.result, instr.line_number)
                    optimized.append(new_instr)
                    self.optimizations_applied.append(f"Algebraic simplification: {instr.arg1} + 0 -> {instr.arg1}")
                    simplified = True
                elif instr.arg1 == '0':
                    new_instr = ThreeAddressCode("ASSIGN", instr.arg2, None, instr.result, instr.line_number)
                    optimized.append(new_instr)
                    self.optimizations_applied.append(f"Algebraic simplification: 0 + {instr.arg2} -> {instr.arg2}")
                    simplified = True
            
            elif instr.operation == '-':
                # x - 0 = x
                if instr.arg2 == '0':
                    new_instr = ThreeAddressCode("ASSIGN", instr.arg1, None, instr.result, instr.line_number)
                    optimized.append(new_instr)
                    self.optimizations_applied.append(f"Algebraic simplification: {instr.arg1} - 0 -> {instr.arg1}")
                    simplified = True
            
            elif instr.operation == '*':
                # x * 1 = x, 1 * x = x, x * 0 = 0, 0 * x = 0
                if instr.arg2 == '1':
                    new_instr = ThreeAddressCode("ASSIGN", instr.arg1, None, instr.result, instr.line_number)
                    optimized.append(new_instr)
                    self.optimizations_applied.append(f"Algebraic simplification: {instr.arg1} * 1 -> {instr.arg1}")
                    simplified = True
                elif instr.arg1 == '1':
                    new_instr = ThreeAddressCode("ASSIGN", instr.arg2, None, instr.result, instr.line_number)
                    optimized.append(new_instr)
                    self.optimizations_applied.append(f"Algebraic simplification: 1 * {instr.arg2} -> {instr.arg2}")
                    simplified = True
                elif instr.arg2 == '0' or instr.arg1 == '0':
                    new_instr = ThreeAddressCode("ASSIGN", "0", None, instr.result, instr.line_number)
                    optimized.append(new_instr)
                    self.optimizations_applied.append(f"Algebraic simplification: multiplication by 0 -> 0")
                    simplified = True
            
            elif instr.operation == '/':
                # x / 1 = x
                if instr.arg2 == '1':
                    new_instr = ThreeAddressCode("ASSIGN", instr.arg1, None, instr.result, instr.line_number)
                    optimized.append(new_instr)
                    self.optimizations_applied.append(f"Algebraic simplification: {instr.arg1} / 1 -> {instr.arg1}")
                    simplified = True
            
            if not simplified:
                optimized.append(instr)
        
        return optimized
    
    def remove_redundant_jumps(self, code: List[ThreeAddressCode]) -> List[ThreeAddressCode]:
        """Remove redundant jumps and unreachable code"""
        optimized = []
        i = 0
        
        while i < len(code):
            instr = code[i]
            
            # Remove jumps to next instruction
            if (instr.operation == "GOTO" and 
                i + 1 < len(code) and 
                code[i + 1].operation == "LABEL" and 
                code[i + 1].result == instr.result):
                
                self.optimizations_applied.append(f"Removed redundant jump to next instruction")
                i += 1
                continue
            
            # Remove unreachable code after unconditional jumps
            if instr.operation in ["GOTO", "RETURN"]:
                optimized.append(instr)
                i += 1
                
                # Skip instructions until we find a label
                while i < len(code) and code[i].operation != "LABEL":
                    self.optimizations_applied.append(f"Removed unreachable code: {code[i].operation}")
                    i += 1
                continue
            
            optimized.append(instr)
            i += 1
        
        return optimized
    
    def is_constant(self, value: str) -> bool:
        """Check if a value is a constant"""
        try:
            float(value)
            return True
        except ValueError:
            return value.lower() in ['true', 'false']
    
    def get_constant_value(self, value: str):
        """Get numeric value of a constant"""
        if value.lower() == 'true':
            return 1
        elif value.lower() == 'false':
            return 0
        else:
            try:
                return int(value)
            except ValueError:
                return float(value)
    
    def is_variable(self, value: str) -> bool:
        """Check if a value is a variable (not a constant)"""
        return not self.is_constant(value) and value and not value.startswith('L')

# =============================================================================
# ASSEMBLY CODE GENERATOR
# =============================================================================

class AssemblyGenerator:
    def __init__(self, target_arch="x86_64"):
        self.target_arch = target_arch
        self.assembly_code = []
        self.data_section = []
        self.text_section = []
        self.register_map = {}
        self.stack_offset = 0
        self.label_map = {}
        self.string_literals = {}
        self.string_counter = 0
        
        # x86_64 registers for parameters
        self.param_registers = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        self.available_registers = ["rax", "rbx", "rcx", "rdx", "r8", "r9", "r10", "r11"]
        self.used_registers = set()
    
    def generate(self, tac_code: List[ThreeAddressCode]) -> str:
        """Generate assembly code from TAC"""
        self.preprocess_strings(tac_code)
        self.generate_data_section()
        self.generate_text_section(tac_code)
        
        return self.assemble_final_code()
    
    def preprocess_strings(self, tac_code: List[ThreeAddressCode]):
        """Extract string literals for data section"""
        for instr in tac_code:
            for arg in [instr.arg1, instr.arg2, instr.result]:
                if arg and self.is_string_literal(arg):
                    if arg not in self.string_literals:
                        self.string_counter += 1
                        self.string_literals[arg] = f"str_{self.string_counter}"
    
    def is_string_literal(self, value: str) -> bool:
        """Check if value is a string literal"""
        return value and len(value) > 1 and value[0] == '"' and value[-1] == '"'
    
    def generate_data_section(self):
        """Generate assembly data section"""
        self.data_section.append(".section .data")
        
        # Add string literals
        for string_val, label in self.string_literals.items():
            escaped_string = string_val[1:-1]  # Remove quotes
            self.data_section.append(f"{label}: .asciz \"{escaped_string}\"")
        
        # Add space for global variables if needed
        self.data_section.append("")
    
    def generate_text_section(self, tac_code: List[ThreeAddressCode]):
        """Generate assembly text section"""
        self.text_section.append(".section .text")
        self.text_section.append(".global _start")
        self.text_section.append("")
        
        current_function = None
        
        for instr in tac_code:
            if instr.operation == "FUNCTION":
                current_function = instr.result
                self.generate_function_prologue(instr.result)
            
            elif instr.operation == "END_FUNCTION":
                self.generate_function_epilogue(instr.result)
                current_function = None
            
            elif instr.operation == "LABEL":
                self.text_section.append(f"{instr.result}:")
            
            elif instr.operation == "GOTO":
                self.text_section.append(f"    jmp {instr.result}")
            
            elif instr.operation == "IF_TRUE":
                self.text_section.append(f"    cmp $0, {self.get_operand(instr.arg1)}")
                self.text_section.append(f"    jne {instr.result}")
            
            elif instr.operation == "IF_FALSE":
                self.text_section.append(f"    cmp $0, {self.get_operand(instr.arg1)}")
                self.text_section.append(f"    je {instr.result}")
            
            elif instr.operation == "ASSIGN":
                self.generate_assignment(instr)
            
            elif instr.operation in ["+", "-", "*", "/", "%"]:
                self.generate_arithmetic(instr)
            
            elif instr.operation in ["==", "!=", "<", ">", "<=", ">="]:
                self.generate_comparison(instr)
            
            elif instr.operation in ["&&", "||"]:
                self.generate_logical(instr)
            
            elif instr.operation == "CALL":
                self.generate_function_call(instr)
            
            elif instr.operation == "RETURN":
                self.generate_return(instr)
            
            elif instr.operation == "DECLARE":
                self.allocate_variable(instr.arg1, instr.result)
            
            elif instr.operation == "DECLARE_ARRAY":
                self.allocate_array(instr.arg1, int(instr.arg2), instr.result)
            
            elif instr.operation == "ARRAY_ACCESS":
                self.generate_array_access(instr)
            
            elif instr.operation == "ARRAY_ASSIGN":
                self.generate_array_assignment(instr)
        
        # Add main entry point if not present
        if "_start" not in [line.split(":")[0] for line in self.text_section if ":" in line]:
            self.text_section.extend([
                "_start:",
                "    call main",
                "    mov $60, %rax    # sys_exit",
                "    mov $0, %rdi     # exit status",
                "    syscall"
            ])
    
    def generate_function_prologue(self, func_name: str):
        """Generate function prologue"""
        if func_name == "main":
            self.text_section.append("main:")
        else:
            self.text_section.append(f"{func_name}:")
        
        self.text_section.extend([
            "    push %rbp",
            "    mov %rsp, %rbp"
        ])
        
        self.stack_offset = 0
    
    def generate_function_epilogue(self, func_name: str):
        """Generate function epilogue"""
        self.text_section.extend([
            "    mov %rbp, %rsp",
            "    pop %rbp",
            "    ret",
            ""
        ])
    
    def generate_assignment(self, instr: ThreeAddressCode):
        """Generate assignment instruction"""
        src = self.get_operand(instr.arg1)
        dst = self.get_location(instr.result)
        
        if src.startswith("$"):  # Immediate value
            self.text_section.append(f"    mov {src}, {dst}")
        else:  # Register or memory
            self.text_section.append(f"    mov {src}, %rax")
            self.text_section.append(f"    mov %rax, {dst}")
    
    def generate_arithmetic(self, instr: ThreeAddressCode):
        """Generate arithmetic operations"""
        arg1 = self.get_operand(instr.arg1)
        arg2 = self.get_operand(instr.arg2)
        result = self.get_location(instr.result)
        
        # Move first operand to rax
        if arg1.startswith("$"):
            self.text_section.append(f"    mov {arg1}, %rax")
        else:
            self.text_section.append(f"    mov {arg1}, %rax")
        
        # Perform operation with second operand
        if instr.operation == "+":
            if arg2.startswith("$"):
                self.text_section.append(f"    add {arg2}, %rax")
            else:
                self.text_section.append(f"    add {arg2}, %rax")
        elif instr.operation == "-":
            if arg2.startswith("$"):
                self.text_section.append(f"    sub {arg2}, %rax")
            else:
                self.text_section.append(f"    sub {arg2}, %rax")
        elif instr.operation == "*":
            if arg2.startswith("$"):
                self.text_section.append(f"    mov {arg2}, %rbx")
                self.text_section.append(f"    imul %rbx, %rax")
            else:
                self.text_section.append(f"    imul {arg2}, %rax")
        elif instr.operation == "/":
            self.text_section.extend([
                "    cqo              # Sign extend rax to rdx:rax",
                f"    mov {arg2}, %rbx",
                "    idiv %rbx"
            ])
        elif instr.operation == "%":
            self.text_section.extend([
                "    cqo              # Sign extend rax to rdx:rax", 
                f"    mov {arg2}, %rbx",
                "    idiv %rbx",
                "    mov %rdx, %rax   # Remainder is in rdx"
            ])
        
        # Store result
        self.text_section.append(f"    mov %rax, {result}")
    
    def generate_comparison(self, instr: ThreeAddressCode):
        """Generate comparison operations"""
        arg1 = self.get_operand(instr.arg1)
        arg2 = self.get_operand(instr.arg2)
        result = self.get_location(instr.result)
        
        # Load operands and compare
        self.text_section.append(f"    mov {arg1}, %rax")
        self.text_section.append(f"    cmp {arg2}, %rax")
        
        # Set result based on comparison
        if instr.operation == "==":
            condition = "sete"
        elif instr.operation == "!=":
            condition = "setne"
        elif instr.operation == "<":
            condition = "setl"
        elif instr.operation == ">":
            condition = "setg"
        elif instr.operation == "<=":
            condition = "setle"
        elif instr.operation == ">=":
            condition = "setge"
        
        self.text_section.extend([
            f"    {condition} %al",
            "    movzx %al, %rax",
            f"    mov %rax, {result}"
        ])
    
    def generate_logical(self, instr: ThreeAddressCode):
        """Generate logical operations"""
        arg1 = self.get_operand(instr.arg1)
        arg2 = self.get_operand(instr.arg2)
        result = self.get_location(instr.result)
        
        if instr.operation == "&&":
            # Short-circuit AND
            false_label = f"and_false_{instr.line_number}"
            end_label = f"and_end_{instr.line_number}"
            
            self.text_section.extend([
                f"    mov {arg1}, %rax",
                "    cmp $0, %rax",
                f"    je {false_label}",
                f"    mov {arg2}, %rax",
                "    cmp $0, %rax",
                f"    je {false_label}",
                "    mov $1, %rax",
                f"    jmp {end_label}",
                f"{false_label}:",
                "    mov $0, %rax",
                f"{end_label}:",
                f"    mov %rax, {result}"
            ])
        
        elif instr.operation == "||":
            # Short-circuit OR
            true_label = f"or_true_{instr.line_number}"
            end_label = f"or_end_{instr.line_number}"
            
            self.text_section.extend([
                f"    mov {arg1}, %rax",
                "    cmp $0, %rax",
                f"    jne {true_label}",
                f"    mov {arg2}, %rax",
                "    cmp $0, %rax",
                f"    jne {true_label}",
                "    mov $0, %rax",
                f"    jmp {end_label}",
                f"{true_label}:",
                "    mov $1, %rax",
                f"{end_label}:",
                f"    mov %rax, {result}"
            ])
    
    def generate_function_call(self, instr: ThreeAddressCode):
        """Generate function call"""
        # For simplicity, assume system V ABI
        self.text_section.append(f"    call {instr.arg1}")
        
        # Store return value
        if instr.result:
            result = self.get_location(instr.result)
            self.text_section.append(f"    mov %rax, {result}")
    
    def generate_return(self, instr: ThreeAddressCode):
        """Generate return statement"""
        if instr.arg1:
            # Load return value into rax
            operand = self.get_operand(instr.arg1)
            self.text_section.append(f"    mov {operand}, %rax")
        
        self.text_section.extend([
            "    mov %rbp, %rsp",
            "    pop %rbp", 
            "    ret"
        ])
    
    def generate_array_access(self, instr: ThreeAddressCode):
        """Generate array access code"""
        array_base = self.get_location(instr.arg1)
        index = self.get_operand(instr.arg2)
        result = self.get_location(instr.result)
        
        # Calculate offset: index * 8 (assuming 8-byte elements)
        self.text_section.extend([
            f"    mov {index}, %rax",
            "    imul $8, %rax",
            f"    add {array_base}, %rax",
            "    mov (%rax), %rbx",
            f"    mov %rbx, {result}"
        ])
    
    def generate_array_assignment(self, instr: ThreeAddressCode):
        """Generate array assignment code"""
        array_base = self.get_location(instr.arg1)
        index = self.get_operand(instr.arg2)
        value = self.get_operand(instr.result)
        
        # Calculate offset and store value
        self.text_section.extend([
            f"    mov {index}, %rax",
            "    imul $8, %rax",
            f"    add {array_base}, %rax",
            f"    mov {value}, %rbx",
            "    mov %rbx, (%rax)"
        ])
    
    def get_operand(self, operand: str) -> str:
        """Get assembly operand (register, memory, or immediate)"""
        if not operand:
            return "$0"
        
        # Check if it's a string literal
        if operand in self.string_literals:
            return f"${self.string_literals[operand]}"
        
        # Check if it's a numeric constant
        try:
            int(operand)
            return f"${operand}"
        except ValueError:
            pass
        
        # Check if it's a boolean constant
        if operand.lower() == "true":
            return "$1"
        elif operand.lower() == "false":
            return "$0"
        
        # It's a variable
        return self.get_location(operand)
    
    def get_location(self, var_name: str) -> str:
        """Get memory location for a variable"""
        if var_name not in self.register_map:
            # Allocate stack space
            self.stack_offset += 8
            self.register_map[var_name] = f"-{self.stack_offset}(%rbp)"
        
        return self.register_map[var_name]
    
    def allocate_variable(self, var_name: str, var_type: str):
        """Allocate space for a variable"""
        if var_name not in self.register_map:
            self.stack_offset += 8
            self.register_map[var_name] = f"-{self.stack_offset}(%rbp)"
    
    def allocate_array(self, array_name: str, size: int, element_type: str):
        """Allocate space for an array"""
        if array_name not in self.register_map:
            # Allocate space for array (size * 8 bytes per element)
            array_size = size * 8
            self.stack_offset += array_size
            self.register_map[array_name] = f"-{self.stack_offset}(%rbp)"
    
    def assemble_final_code(self) -> str:
        """Assemble final assembly code"""
        assembly_lines = []
        
        # Add data section
        assembly_lines.extend(self.data_section)
        assembly_lines.append("")
        
        # Add text section
        assembly_lines.extend(self.text_section)
        
        return "\n".join(assembly_lines)

# =============================================================================
# ENHANCED MAIN COMPILER CLASS
# =============================================================================

class MiniLangCompiler:
    def __init__(self):
        self.lexer = None
        self.parser = None
        self.semantic_analyzer = None
        self.code_generator = None
        self.optimizer = None
        self.assembly_generator = None
        
        self.tokens = []
        self.ast = None
        self.symbol_table = None
        self.intermediate_code = []
        self.optimized_code = []
        self.assembly_code = ""
        
        self.lexical_errors = []
        self.syntax_errors = []
        self.semantic_errors = []
    
    def compile(self, source_code: str, optimize: bool = True, target_arch: str = "x86_64") -> Dict[str, Any]:
        """
        Complete compilation process
        Returns a dictionary with compilation results
        """
        results = {
            'success': False,
            'tokens': [],
            'ast': None,
            'symbol_table': None,
            'intermediate_code': [],
            'optimized_code': [],
            'assembly_code': "",
            'optimizations_applied': [],
            'errors': {
                'lexical': [],
                'syntax': [],
                'semantic': []
            },
            'statistics': {
                'lines_of_code': len(source_code.split('\n')),
                'tokens_count': 0,
                'functions_count': 0,
                'variables_count': 0,
                'optimizations_count': 0
            }
        }
        
        try:
            # Lexical Analysis
            self.lexer = LexicalAnalyzer(source_code)
            self.tokens = self.lexer.tokenize()
            self.lexical_errors = self.lexer.errors
            
            # Convert tokens to JSON-serializable format
            results['tokens'] = [
                {
                    'type': token.type.value,
                    'value': token.value,
                    'line': token.line,
                    'column': token.column
                }
                for token in self.tokens if token.type != TokenType.EOF
            ]
            
            results['statistics']['tokens_count'] = len(results['tokens'])
            results['errors']['lexical'] = [str(e) for e in self.lexical_errors]
            
            if self.lexical_errors:
                return results
            
            # Syntax Analysis
            self.parser = Parser(self.tokens)
            self.ast = self.parser.parse()
            self.syntax_errors = self.parser.errors
            
            results['ast'] = self.ast_to_dict(self.ast)
            results['errors']['syntax'] = [str(e) for e in self.syntax_errors]
            
            if self.syntax_errors:
                return results
            
            # Count functions
            results['statistics']['functions_count'] = len(self.ast.functions)
            
            # Semantic Analysis
            self.semantic_analyzer = SemanticAnalyzer()
            self.semantic_analyzer.set_functions(self.ast.functions)
            self.semantic_analyzer.analyze(self.ast)
            self.semantic_errors = self.semantic_analyzer.errors

            results['symbol_table'] = self.symbol_table_to_dict()
            results['errors']['semantic'] = [str(e) for e in self.semantic_errors]

            results['statistics']['variables_count'] = len([ 
                s for scope in self.semantic_analyzer.symbol_table.all_symbols 
                for s in [scope] if s.kind in ['variable', 'parameter']
            ])
            
            # Get unused variables as warnings, not errors
            unused_variables = self.semantic_analyzer.get_unused_variables()
            if unused_variables:
                results['warnings'] = [f"Unused variable: {var}" for var in unused_variables]
            else:
                results['warnings'] = []
            
            if self.semantic_errors:
                return results
            
            # Intermediate Code Generation
            self.code_generator = IntermediateCodeGenerator()
            self.intermediate_code = self.code_generator.generate(self.ast)
            
            results['intermediate_code'] = [
                {
                    'operation': tac.operation,
                    'arg1': tac.arg1,
                    'arg2': tac.arg2,
                    'result': tac.result,
                    'line_number': tac.line_number
                }
                for tac in self.intermediate_code
            ]
            
            # Code Optimization (if enabled)
            if optimize:
                self.optimizer = CodeOptimizer()
                self.optimized_code = self.optimizer.optimize(self.intermediate_code)
                results['optimizations_applied'] = self.optimizer.optimizations_applied
                results['statistics']['optimizations_count'] = len(self.optimizer.optimizations_applied)
            else:
                self.optimized_code = self.intermediate_code
            
            results['optimized_code'] = [
                {
                    'operation': tac.operation,
                    'arg1': tac.arg1,
                    'arg2': tac.arg2,
                    'result': tac.result,
                    'line_number': tac.line_number
                }
                for tac in self.optimized_code
            ]
            
            # Assembly Code Generation
            self.assembly_generator = AssemblyGenerator(target_arch)
            self.assembly_code = self.assembly_generator.generate(self.optimized_code)
            results['assembly_code'] = self.assembly_code
            
            results['success'] = True
            
        except Exception as e:
            results['errors']['general'] = [f"Compilation error: {str(e)}"]
        
        return results
    
    def ast_to_dict(self, node) -> dict:
        """Convert AST node to dictionary for JSON serialization"""
        if node is None:
            return None
        
        result = {'type': type(node).__name__}
        
        if isinstance(node, Program):
            result['functions'] = [self.ast_to_dict(func) for func in node.functions]
        
        elif isinstance(node, Function):
            result['name'] = node.name
            result['return_type'] = node.return_type
            result['parameters'] = [self.ast_to_dict(param) for param in node.parameters]
            result['body'] = [self.ast_to_dict(stmt) for stmt in node.body]
        
        elif isinstance(node, Parameter):
            result['name'] = node.name
            result['type'] = node.type
            result['is_array'] = node.is_array
            result['array_size'] = node.array_size
        
        elif isinstance(node, VarDeclaration):
            result['name'] = node.name
            result['type'] = node.type
            result['is_array'] = node.is_array
            result['array_size'] = node.array_size
            result['initializer'] = self.ast_to_dict(node.initializer)
        
        elif isinstance(node, Assignment):
            result['target'] = self.ast_to_dict(node.target)
            result['value'] = self.ast_to_dict(node.value)
        
        elif isinstance(node, IfStatement):
            result['condition'] = self.ast_to_dict(node.condition)
            result['then_stmt'] = self.ast_to_dict(node.then_stmt)
            result['else_stmt'] = self.ast_to_dict(node.else_stmt)
        
        elif isinstance(node, WhileStatement):
            result['condition'] = self.ast_to_dict(node.condition)
            result['body'] = self.ast_to_dict(node.body)
        
        elif isinstance(node, ForStatement):
            result['init'] = self.ast_to_dict(node.init)
            result['condition'] = self.ast_to_dict(node.condition)
            result['update'] = self.ast_to_dict(node.update)
            result['body'] = self.ast_to_dict(node.body)
        
        elif isinstance(node, ReturnStatement):
            result['value'] = self.ast_to_dict(node.value)
        
        elif isinstance(node, BreakStatement):
            pass  # No additional fields
        
        elif isinstance(node, ContinueStatement):
            pass  # No additional fields
        
        elif isinstance(node, Block):
            result['statements'] = [self.ast_to_dict(stmt) for stmt in node.statements]
        
        elif isinstance(node, ExpressionStatement):
            result['expression'] = self.ast_to_dict(node.expression)
        
        elif isinstance(node, BinaryExpression):
            result['left'] = self.ast_to_dict(node.left)
            result['operator'] = node.operator
            result['right'] = self.ast_to_dict(node.right)
        
        elif isinstance(node, UnaryExpression):
            result['operator'] = node.operator
            result['operand'] = self.ast_to_dict(node.operand)
            result['is_postfix'] = node.is_postfix
        
        elif isinstance(node, FunctionCall):
            result['name'] = node.name
            result['arguments'] = [self.ast_to_dict(arg) for arg in node.arguments]
        
        elif isinstance(node, ArrayAccess):
            result['array'] = self.ast_to_dict(node.array)
            result['index'] = self.ast_to_dict(node.index)
        
        elif isinstance(node, Identifier):
            result['name'] = node.name
        
        elif isinstance(node, Literal):
            result['value'] = node.value
            result['value_type'] = node.type
        
        return result
    
    def symbol_table_to_dict(self):
        """Convert symbol table to dictionary for JSON serialization"""
        if not self.semantic_analyzer or not self.semantic_analyzer.symbol_table:
            return {}
        
        result = {}
        symbol_table = self.semantic_analyzer.symbol_table
        
        # Include all scopes
        for scope_level, scope in enumerate(symbol_table.scopes):
            scope_name = f"scope_{scope_level}"
            scope_info = {
                'level': scope_level,
                'symbols': {},
                'description': self.get_scope_description(scope_level)
            }
            
            for symbol_name, symbol in scope.items():
                scope_info['symbols'][symbol_name] = {
                    "name": symbol.name,
                    "type": symbol.type,
                    "kind": symbol.kind,
                    "scope_level": symbol.scope_level,
                    "initialized": symbol.initialized,
                    "is_array": symbol.is_array,
                    "array_size": symbol.array_size,
                    "used": symbol.used
                }
            
            result[scope_name] = scope_info
        
        return result
    
    def get_scope_description(self, level):
        """Get human-readable description for scope level"""
        if level == 0:
            return "Global Scope"
        elif level == 1:
            return "Function Scope"
        elif level >= 2:
            return f"Block Scope (Level {level})"
        else:
            return f"Scope Level {level}"
    
    def print_compilation_report(self, results: Dict[str, Any]):
        """Print detailed compilation report"""
        print("=" * 70)
        print("MINILANG++ COMPILATION REPORT")
        print("=" * 70)
        
        stats = results.get('statistics', {})
        print(f"Lines of Code: {stats.get('lines_of_code', 0)}")
        print(f"Tokens: {stats.get('tokens_count', 0)}")
        print(f"Functions: {stats.get('functions_count', 0)}")
        print(f"Variables: {stats.get('variables_count', 0)}")
        
        if results['success']:
            print("✅ Compilation Status: SUCCESS")
            
            if stats.get('optimizations_count', 0) > 0:
                print(f"🔧 Optimizations Applied: {stats['optimizations_count']}")
                for opt in results.get('optimizations_applied', [])[:5]:  # Show first 5
                    print(f"   • {opt}")
                if len(results.get('optimizations_applied', [])) > 5:
                    print(f"   ... and {len(results['optimizations_applied']) - 5} more")
            
            print(f"\n📊 Code Statistics:")
            print(f"   Intermediate Instructions: {len(results.get('intermediate_code', []))}")
            print(f"   Optimized Instructions: {len(results.get('optimized_code', []))}")
            print(f"   Assembly Lines: {len(results.get('assembly_code', '').split())}")
            
        else:
            print("❌ Compilation Status: FAILED")
            
            # Print errors
            for error_type, errors in results.get('errors', {}).items():
                if errors:
                    print(f"\n🚨 {error_type.upper()} ERRORS:")
                    for error in errors:
                        print(f"   • {error}")



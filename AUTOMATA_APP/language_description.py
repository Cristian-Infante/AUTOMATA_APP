# language_description.py

import re
from itertools import count


# --- Clases para el AST (Árbol Sintáctico Abstracto) ---

class RegexNode:
    """Clase base para todos los nodos del AST."""
    pass


class UnionNode(RegexNode):
    """Nodo que representa una unión (operador |)."""

    def __init__(self, alternatives):
        self.alternatives = alternatives  # Lista de RegexNode


class ConcatNode(RegexNode):
    """Nodo que representa una concatenación de términos."""

    def __init__(self, terms):
        self.terms = terms  # Lista de RegexNode


class RepetitionNode(RegexNode):
    """Nodo que representa una repetición (*, +, ?)."""

    def __init__(self, term, operator):
        self.term = term  # RegexNode
        self.operator = operator  # '*', '+', '?'


class CharacterNode(RegexNode):
    """Nodo que representa un carácter individual."""

    def __init__(self, character):
        self.character = character


# --- Función de Tokenización ---

def tokenize_regex(s):
    """
    Tokeniza la expresión regular en una lista de tokens.

    Args:
        s (str): Expresión regular.

    Returns:
        list: Lista de tokens.
    """
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in '()*+?|':
            tokens.append(c)
            i += 1
        elif c.isalnum():
            tokens.append(c)
            i += 1
        elif c == '\\':
            # Secuencia de escape
            if i + 1 < len(s):
                tokens.append(s[i + 1])
                i += 2
            else:
                raise ValueError("Secuencia de escape inválida al final de la expresión regular")
        else:
            i += 1  # Ignorar otros caracteres (p. ej., espacios)
    return tokens


# --- Parser Recursivo ---

class Parser:
    """
    Parser recursivo para construir el AST a partir de tokens.
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        """
        Retorna el token actual.
        """
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, token=None):
        """
        Consume el token actual si coincide con el esperado.

        Args:
            token (str, optional): Token esperado. Si es None, consume cualquier token.

        Returns:
            str: Token consumido.
        """
        current = self.current_token()
        if token is None or current == token:
            self.pos += 1
            return current
        else:
            raise ValueError(f"Se esperaba '{token}', pero se encontró '{current}'")

    def parse_regex(self):
        """
        Punto de entrada para el parser.

        Returns:
            RegexNode: Nodo raíz del AST.
        """
        return self.parse_union()

    def parse_union(self):
        """
        Parsea una expresión con posibles uniones (|).

        Returns:
            RegexNode: Nodo de unión o el término sin unión.
        """
        left = self.parse_concatenation()
        alternatives = [left]
        while self.current_token() == '|':
            self.eat('|')
            right = self.parse_concatenation()
            alternatives.append(right)
        if len(alternatives) > 1:
            return UnionNode(alternatives)
        else:
            return left

    def parse_concatenation(self):
        """
        Parsea una concatenación de términos.

        Returns:
            RegexNode: Nodo de concatenación o el término individual.
        """
        terms = []
        while True:
            term = self.parse_repetition()
            if term is not None:
                terms.append(term)
            else:
                break
            # Verificar si el siguiente token puede iniciar un nuevo término
            next_token = self.current_token()
            if next_token in (')', '|', None):
                break
        if len(terms) == 1:
            return terms[0]
        elif len(terms) > 1:
            return ConcatNode(terms)
        else:
            return None

    def parse_repetition(self):
        """
        Parsea un término con posibles operadores de repetición (*, +, ?).

        Returns:
            RegexNode: Nodo de repetición o el término individual.
        """
        term = self.parse_atom()
        if term is not None:
            token = self.current_token()
            if token in ('*', '+', '?'):
                operator = self.eat()
                return RepetitionNode(term, operator)
            else:
                return term
        else:
            return None

    def parse_atom(self):
        """
        Parsea un átomo: un carácter o una subexpresión entre paréntesis.

        Returns:
            RegexNode: Nodo de carácter o el nodo de la subexpresión.
        """
        token = self.current_token()
        if token == '(':
            self.eat('(')
            node = self.parse_regex()
            self.eat(')')
            return node
        elif token and token.isalnum():
            return CharacterNode(self.eat())
        else:
            return None


def parse_regex(s):
    """
    Función auxiliar para parsear una expresión regular y obtener el AST.

    Args:
        s (str): Expresión regular.

    Returns:
        RegexNode: Nodo raíz del AST.
    """
    tokens = tokenize_regex(s)
    parser = Parser(tokens)
    ast = parser.parse_regex()
    return ast


# --- Generación de la Descripción Formal ---

def generate_description_from_ast(node):
    """
    Genera una descripción formal en LaTeX a partir del AST.

    Args:
        node (RegexNode): Nodo del AST.

    Returns:
        str: Descripción formal en LaTeX.
    """
    if isinstance(node, UnionNode):
        alternatives = [generate_description_from_ast(alt) for alt in node.alternatives]
        return ' | '.join(alternatives)
    elif isinstance(node, ConcatNode):
        # Generar descripción para cada término y concatenarlos
        terms = [generate_description_from_ast(term) for term in node.terms]
        return ''.join(terms)
    elif isinstance(node, RepetitionNode):
        term_desc = generate_description_from_ast(node.term)
        if node.operator == '*':
            return f"({term_desc})^*"
        elif node.operator == '+':
            return f"({term_desc})^+"
        elif node.operator == '?':
            return f"({term_desc})?"
    elif isinstance(node, CharacterNode):
        return node.character
    else:
        return ''


def detect_specific_patterns(node):
    """
    Detecta patrones específicos en el AST para generar descripciones más precisas.

    Args:
        node (RegexNode): Nodo del AST.

    Returns:
        str or None: Descripción específica o None si no se detecta.
    """
    variables = ['m', 'n', 'o', 'p', 'q', 'r', 's', 't']
    var_generator = iter(variables)
    var_dict = {}  # Mapping from RepetitionNode to variable

    # Helper function to traverse the AST and assign variables
    def traverse_and_assign(node):
        if isinstance(node, RepetitionNode):
            try:
                var = next(var_generator)
            except StopIteration:
                raise ValueError("Se excedió el número de variables disponibles para asignar a las repeticiones.")
            var_dict[node] = var
            traverse_and_assign(node.term)
        elif isinstance(node, UnionNode):
            for alt in node.alternatives:
                traverse_and_assign(alt)
        elif isinstance(node, ConcatNode):
            for term in node.terms:
                traverse_and_assign(term)
        # CharacterNode doesn't need traversal

    traverse_and_assign(node)

    # Now, build the expression and conditions
    expression = reconstruct_expression(node, var_dict)
    conditions = []

    # Collect conditions based on the assigned variables
    for repetition_node, var in var_dict.items():
        if repetition_node.operator == '*':
            conditions.append(f"{var} \\geq 0")
        elif repetition_node.operator == '+':
            conditions.append(f"{var} > 0")
        elif repetition_node.operator == '?':
            conditions.append(f"{var} \\in \\{{0,1\\}}")

    if expression and conditions:
        conditions_str = ', '.join(conditions)
        return f"{expression}, {conditions_str}"
    elif expression:
        return f"{expression}"
    else:
        return None


def reconstruct_expression(node, var_dict):
    """
    Reconstruye la expresión regular con variables asignadas a las repeticiones y añade paréntesis donde es necesario.

    Args:
        node (RegexNode): Nodo del AST.
        var_dict (dict): Mapeo de RepetitionNode a variables.

    Returns:
        str: Expresión con variables y condiciones.
    """
    expression = ""

    if isinstance(node, UnionNode):
        # Reconstruye la unión con espacios alrededor del operador |
        inner = ' | '.join([reconstruct_expression(alt, var_dict) for alt in node.alternatives])
        # Siempre añadir paréntesis alrededor de una unión
        expression = f"({inner})"
    elif isinstance(node, ConcatNode):
        # Reconstruye la concatenación de términos
        parts = []
        for term in node.terms:
            part = reconstruct_expression(term, var_dict)
            parts.append(part)
        expression = ''.join(parts)
    elif isinstance(node, RepetitionNode):
        var = var_dict.get(node)
        if var:
            if isinstance(node.term, UnionNode):
                # Si el término es una unión, ya está entre paréntesis
                inner = reconstruct_expression(node.term, var_dict)
                expression = f"{inner}^{var}"
            else:
                # Si no es una unión, no es necesario añadir paréntesis
                inner = reconstruct_expression(node.term, var_dict)
                expression = f"{inner}^{var}"
        else:
            # Si no hay variable asignada, usar la descripción por defecto
            expression = generate_description_from_ast(node)
    elif isinstance(node, CharacterNode):
        # Retorna el carácter directamente
        expression = node.character
    return expression


def generate_language_description(regex):
    """
    Genera una descripción formal del lenguaje aceptado por una expresión regular.

    Args:
        regex (str): Expresión regular.

    Returns:
        str: Descripción formal en LaTeX.
    """
    try:
        # Limpiar la expresión regular
        regex = regex.replace(' ', '')
        # Crear el AST
        ast = parse_regex(regex)
        # Detectar patrones específicos para generar descripciones más precisas
        pattern_specific = detect_specific_patterns(ast)
        if pattern_specific:
            # Usar una f-string cruda (raw f-string) para evitar problemas con las llaves
            return rf"L = \{{ {pattern_specific} \}}"

        # Si no se reconoce un patrón específico, usar una descripción genérica
        # Definir el alfabeto (Σ)
        alphabet = set(re.findall(r'\w', regex))
        alphabet_str = ', '.join(sorted(alphabet))
        return rf"L = \{{ x \mid x \in \{{{alphabet_str}}}^* \text{{ y cumple la expresión regular }} \texttt{{ {regex} }} \}}"
    except Exception as e:
        # En caso de error, devolver una descripción genérica
        return rf"L = \{{ x \in \Sigma^* \mid x \text{{ cumple la expresión regular }} \texttt{{ {regex} }} \}}"

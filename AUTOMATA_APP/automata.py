import re
import base64
import string
from graphviz import Digraph

class NFA:
    def __init__(self, start_state, accept_states):
        self.start_state = start_state
        self.accept_states = accept_states

class State:
    _id = 0

    def __init__(self, is_final=False, is_initial=False):
        self.name = f'S{State._id}'
        State._id += 1
        self.transitions = {}  # Regular transitions
        self.epsilon_transitions = set()  # Epsilon transitions (λ)
        self.is_final = is_final
        self.is_initial = is_initial

    def add_transition(self, symbol, state):
        if symbol == 'λ':
            # Agregar una transición lambda
            self.epsilon_transitions.add(state)
        else:
            # Asegurarse de que se fusionen correctamente las transiciones existentes
            if symbol not in self.transitions:
                self.transitions[symbol] = set()
            self.transitions[symbol].add(state)

    @staticmethod
    def reset_state_counter():
        """
        Reinicia el contador global de IDs de estados.
        """
        State._id = 0

def reset_state_counter():
    State._id = 0

class NFA:
    def __init__(self, start_state, accept_states):
        self.start_state = start_state
        self.accept_states = accept_states

def tokenize(regex):
    tokens = []
    i = 0
    length = len(regex)
    while i < length:
        char = regex[i]
        if char.isalnum():
            tokens.append(('symbol', char))
            i += 1
        elif char == '(':
            tokens.append(('left_paren', char))
            i += 1
        elif char == ')':
            tokens.append(('right_paren', char))
            i += 1
        elif char in ['*', '?']:
            tokens.append(('unary_operator', char))
            i += 1
        elif char == '+':
            # Determinar si '+' es binario o unario
            prev_char = regex[i - 1] if i > 0 else None
            next_char = regex[i + 1] if i + 1 < length else None

            if prev_char and prev_char not in ['(', '|'] and \
               (next_char is None or next_char in [')', '|', '+', '*', '?']):
                # Si el carácter anterior es símbolo, ')' o operador unario,
                # y el siguiente es fin de cadena, ')', operador binario o unario,
                # entonces '+' es un operador unario
                tokens.append(('unary_operator', char))
            else:
                # En otros casos, '+' es un operador binario
                tokens.append(('binary_operator', char))
            i += 1
        elif char == '|':
            tokens.append(('binary_operator', char))
            i += 1
        else:
            # Manejar otros caracteres o espacios en blanco
            i += 1
    return tokens

def insert_concatenation(tokens):
    result = []
    num_tokens = len(tokens)
    for i in range(num_tokens - 1):
        token = tokens[i]
        next_token = tokens[i + 1]
        result.append(token)
        # No insertar concatenación entre 'right_paren' y 'unary_operator'
        if (token[0] in ['symbol', 'unary_operator', 'right_paren']) and \
           (next_token[0] in ['symbol', 'left_paren']):

            result.append(('binary_operator', '.'))
    result.append(tokens[-1])
    return result

precedence = {'*': 5, '+': 5, '?': 5, '.': 4, '|': 3}

def infix_to_postfix(tokens):
    output = []
    stack = []

    for token in tokens:
        if token[0] == 'symbol':
            output.append(token)
        elif token[0] == 'left_paren':
            stack.append(token)
        elif token[0] == 'right_paren':
            while stack and stack[-1][0] != 'left_paren':
                output.append(stack.pop())
            stack.pop()  # Remove '('
        elif token[0] == 'unary_operator':
            while stack and stack[-1][0] != 'left_paren' and \
                    precedence[token[1]] > precedence.get(stack[-1][1], 0):
                output.append(stack.pop())
            stack.append(token)
        elif token[0] == 'binary_operator':
            while stack and stack[-1][0] != 'left_paren' and \
                    precedence[token[1]] <= precedence.get(stack[-1][1], 0):
                output.append(stack.pop())
            stack.append(token)
        else:
            pass

    while stack:
        output.append(stack.pop())

    return output

def regex_to_nfa(regex):
    tokens = tokenize(regex)
    tokens = insert_concatenation(tokens)
    postfix_tokens = infix_to_postfix(tokens)
    stack = []

    for token in postfix_tokens:
        if token[0] == 'symbol':
            start_state = State(is_initial=True)
            accept_state = State(is_final=True)
            start_state.add_transition(token[1], accept_state)
            nfa = NFA(start_state, [accept_state])
            stack.append(nfa)
        elif token[0] == 'unary_operator':
            if token[1] == '*':
                nfa = stack.pop()
                start_state = State(is_initial=True)
                accept_state = State(is_final=True)

                start_state.add_transition('λ', nfa.start_state)
                start_state.add_transition('λ', accept_state)

                for a_state in nfa.accept_states:
                    a_state.add_transition('λ', nfa.start_state)
                    a_state.add_transition('λ', accept_state)
                    a_state.is_final = False

                nfa = NFA(start_state, [accept_state])
                stack.append(nfa)
            elif token[1] == '+':
                nfa = stack.pop()
                start_state = State(is_initial=True)
                accept_state = State(is_final=True)

                start_state.add_transition('λ', nfa.start_state)

                for a_state in nfa.accept_states:
                    a_state.add_transition('λ', nfa.start_state)
                    a_state.add_transition('λ', accept_state)
                    a_state.is_final = False

                nfa = NFA(start_state, [accept_state])
                stack.append(nfa)
            elif token[1] == '?':
                nfa = stack.pop()
                start_state = State(is_initial=True)
                accept_state = State(is_final=True)

                start_state.add_transition('λ', nfa.start_state)
                start_state.add_transition('λ', accept_state)

                for a_state in nfa.accept_states:
                    a_state.add_transition('λ', accept_state)
                    a_state.is_final = False

                nfa = NFA(start_state, [accept_state])
                stack.append(nfa)
        elif token[0] == 'binary_operator':
            if token[1] == '|':
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                start_state = State(is_initial=True)
                accept_state = State(is_final=True)

                start_state.add_transition('λ', nfa1.start_state)
                start_state.add_transition('λ', nfa2.start_state)

                for a_state in nfa1.accept_states:
                    a_state.add_transition('λ', accept_state)
                    a_state.is_final = False
                for a_state in nfa2.accept_states:
                    a_state.add_transition('λ', accept_state)
                    a_state.is_final = False

                nfa = NFA(start_state, [accept_state])
                stack.append(nfa)
            elif token[1] == '+':
                # Treat '+' as alternation operator
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                start_state = State(is_initial=True)
                accept_state = State(is_final=True)

                start_state.add_transition('λ', nfa1.start_state)
                start_state.add_transition('λ', nfa2.start_state)

                for a_state in nfa1.accept_states:
                    a_state.add_transition('λ', accept_state)
                    a_state.is_final = False
                for a_state in nfa2.accept_states:
                    a_state.add_transition('λ', accept_state)
                    a_state.is_final = False

                nfa = NFA(start_state, [accept_state])
                stack.append(nfa)
            elif token[1] == '.':
                nfa2 = stack.pop()
                nfa1 = stack.pop()

                for a_state in nfa1.accept_states:
                    a_state.add_transition('λ', nfa2.start_state)
                    a_state.is_final = False

                nfa = NFA(nfa1.start_state, nfa2.accept_states)
                nfa.start_state.is_initial = True
                stack.append(nfa)
        else:
            # Handle invalid tokens if necessary
            pass

    final_nfa = stack.pop()
    final_nfa.start_state.is_initial = True
    return final_nfa

def remove_lambda_transitions(nfa):
    new_states = {}
    start_state = None

    old_states = get_all_states(nfa.start_state)

    for old_state in old_states:
        new_state = State(is_initial=old_state.is_initial)
        new_states[old_state] = new_state
        if old_state == nfa.start_state:
            start_state = new_state

    for old_state in old_states:
        new_state = new_states[old_state]
        epsilon_closure = get_epsilon_closure(old_state)

        if any(state.is_final for state in epsilon_closure):
            new_state.is_final = True

        for e_state in epsilon_closure:
            for symbol, next_states in e_state.transitions.items():
                if symbol != 'λ':
                    for next_state in next_states:
                        new_state.add_transition(symbol, new_states[next_state])

    accept_states = [state for state in new_states.values() if state.is_final]

    return NFA(start_state, accept_states)

def get_epsilon_closure(state, closure=None):
    if closure is None:
        closure = set()
    closure.add(state)
    for e_state in state.epsilon_transitions:
        if e_state not in closure:
            get_epsilon_closure(e_state, closure)
    return closure

def get_all_states(start_state, states=None):
    if states is None:
        states = set()
    states.add(start_state)
    for symbol, next_states in start_state.transitions.items():
        for state in next_states:
            if state not in states:
                get_all_states(state, states)
    for e_state in start_state.epsilon_transitions:
        if e_state not in states:
            get_all_states(e_state, states)
    return states

def nfa_to_dfa(nfa):
    start_closure = get_epsilon_closure(nfa.start_state)
    dfa_states = {}
    unmarked_states = []

    start_state = frozenset(start_closure)
    dfa_states[start_state] = State()
    unmarked_states.append(start_state)

    while unmarked_states:
        current = unmarked_states.pop()
        current_state = dfa_states[current]

        symbols = set()
        for nfa_state in current:
            symbols.update(nfa_state.transitions.keys())

        symbols.discard('λ')

        for symbol in symbols:
            next_states = set()
            for nfa_state in current:
                if symbol in nfa_state.transitions:
                    for next_state in nfa_state.transitions[symbol]:
                        next_states.update(get_epsilon_closure(next_state))

            next_state_frozen = frozenset(next_states)
            if next_state_frozen not in dfa_states:
                dfa_states[next_state_frozen] = State()
                unmarked_states.append(next_state_frozen)

            current_state.add_transition(symbol, dfa_states[next_state_frozen])

    for state_set, dfa_state in dfa_states.items():
        for nfa_accept_state in nfa.accept_states:
            if nfa_accept_state in state_set:
                dfa_state.is_final = True
                break

    dfa = NFA(dfa_states[start_state], [s for s in dfa_states.values() if s.is_final])
    return dfa

def visualize_automata(automata):
    dot = Digraph()

    visited = set()
    stack = [automata.start_state]

    while stack:
        state = stack.pop()
        if state.name in visited:
            continue
        visited.add(state.name)

        shape = 'doublecircle' if state.is_final else 'circle'
        dot.node(state.name, shape=shape)

        for symbol, states in state.transitions.items():
            for s in states:
                dot.edge(state.name, s.name, label=symbol)
                if s.name not in visited:
                    stack.append(s)

        for s in state.epsilon_transitions:
            dot.edge(state.name, s.name, label='λ', style='dashed')
            if s.name not in visited:
                stack.append(s)

    dot.attr('node', shape='none')
    dot.node('')
    dot.edge('', automata.start_state.name)

    img = dot.pipe(format='png')
    encoded = base64.b64encode(img).decode('utf-8')
    return encoded

def index_to_letter(index):
    letters = ''
    while True:
        index, remainder = divmod(index, 26)
        letters = chr(65 + remainder) + letters
        if index == 0:
            break
        index -= 1
    return letters

def regex_to_grammar(nfa):
    # Obtener todos los estados del NFA
    states = get_all_states(nfa.start_state)
    states = list(states)

    # Crear un mapeo de estados a símbolos de variables usando letras
    state_symbols = {state: index_to_letter(index) for index, state in enumerate(states)}

    # La variable inicial será la correspondiente al estado inicial
    initial_variable = state_symbols[nfa.start_state]

    # Conjuntos N (no terminales) y T (terminales)
    N = set()
    T = set()
    P = []  # Lista de producciones en forma de tuplas (variable, producción)

    for state in states:
        variable = state_symbols[state]
        N.add(variable)
        productions = []

        # Transiciones normales
        for symbol, next_states in state.transitions.items():
            T.add(symbol)  # Agregar al conjunto de terminales
            for next_state in next_states:
                next_variable = state_symbols[next_state]
                production = f'{symbol}{next_variable}'
                productions.append(production)
                P.append((variable, production))

        # Transiciones epsilon
        for next_state in state.epsilon_transitions:
            next_variable = state_symbols[next_state]
            productions.append(next_variable)
            P.append((variable, next_variable))

        # Si es estado final, agregar producción a ε (cadena vacía)
        if state.is_final:
            productions.append('λ')
            P.append((variable, 'λ  '))

    # Convertir N y T a listas ordenadas
    N = sorted(N)
    T = sorted(T)

    return initial_variable, N, T, P

def simulate_dfa(dfa, input_string):
    current_state = dfa.start_state
    for symbol in input_string:
        if symbol in current_state.transitions:
            current_state = next(iter(current_state.transitions[symbol]))
        else:
            return False
    return current_state.is_final

def dfa_to_grammar(dfa):
    # Obtener todos los estados del DFA
    states = get_all_states(dfa.start_state)
    N = set(state.name for state in states)
    T = set()
    P = []

    for state in states:
        variable = state.name
        for symbol, next_states in state.transitions.items():
            T.add(symbol)
            for next_state in next_states:
                production = (variable, symbol + next_state.name)
                P.append(production)

        # Si el estado es final, agregar producción a λ
        if state.is_final:
            P.append((variable, 'λ'))

    initial_variable = dfa.start_state.name

    return initial_variable, N, T, P
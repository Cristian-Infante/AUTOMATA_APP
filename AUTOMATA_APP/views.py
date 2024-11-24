import traceback
from datetime import datetime
from lib2to3.fixes.fix_input import context

from django.http import JsonResponse
from django.shortcuts import render
from .forms import RegexForm
from .automata import regex_to_nfa, remove_lambda_transitions, nfa_to_dfa, visualize_automata, reset_state_counter, \
    regex_to_grammar, simulate_dfa, dfa_to_grammar, State, NFA
from . import automata
from .language_description import generate_language_description, determine_grammar_type
import json

def index(request):
    current_year = datetime.now().year
    return render(request, 'index.html', {'current_year': current_year})

def regex_to_automata(request):
    if request.method == 'POST':
        form = RegexForm(request.POST)
        if form.is_valid():
            regex = form.cleaned_data['regex']
            State.reset_state_counter()
            reset_state_counter()
            nfa_lambda = regex_to_nfa(regex)
            State.reset_state_counter()
            reset_state_counter()
            nfa_no_lambda = remove_lambda_transitions(nfa_lambda)
            State.reset_state_counter()
            reset_state_counter()
            dfa = nfa_to_dfa(nfa_no_lambda)

            # Generar gráficos
            nfa_lambda_graph = visualize_automata(nfa_lambda)
            nfa_no_lambda_graph = visualize_automata(nfa_no_lambda)
            dfa_graph = visualize_automata(dfa)

            automata_list = [
                ('AFND-λ', nfa_lambda_graph, 'nfa_lambda'),
                ('AFND', nfa_no_lambda_graph, 'nfa_no_lambda'),
                ('AFD', dfa_graph, 'dfa')
            ]

            # Generar la gramática utilizando el NFA existente
            initial_variable, N, T, P = regex_to_grammar(nfa_lambda)

            # Manejar cadenas de prueba
            test_strings = request.POST.getlist('test_strings')

            test_results = []
            for s in test_strings:
                s = s.strip()
                if s == '':
                    continue
                is_accepted = simulate_dfa(dfa, s)
                test_results.append((s, is_accepted))

            # Agrupar producciones por variable
            productions_grouped = {}
            for variable, prod in P:
                if variable not in productions_grouped:
                    productions_grouped[variable] = []
                productions_grouped[variable].append(prod)

            # Preparar la expresión formal del lenguaje aceptado
            language_description = generate_language_description(regex)
            grammar_type = determine_grammar_type(N, T, P)

            context = {
                'form': form,
                'automata_list': automata_list,
                'initial_variable': initial_variable,
                'non_terminals': N,
                'terminals': T,
                'productions_grouped': productions_grouped,
                'language_description': language_description,
                'test_results': test_results,
                'grammar_type': grammar_type
            }
            return render(request, 'regex_to_automata.html', context)
    else:
        form = RegexForm()
    return render(request, 'regex_to_automata.html', {'form': form})

def draw_automata(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            automata.State.reset_state_counter()

            automata_data = data.get('automata')
            state_id_map = data.get('stateIdMap')

            # Procesar el autómata
            states = set()
            input_symbols = set()
            transitions = {}
            initial_state = None
            final_states = set()

            # Mapear los IDs a los nombres de los estados
            id_to_state = { state_id: state_name for state_id, state_name in state_id_map.items() }

            # Construir el conjunto de estados
            for state_id in automata_data['states']:
                state_name = id_to_state[state_id]
                states.add(state_name)

            # Construir las transiciones
            for transition in automata_data['transitions']:
                source = id_to_state[transition['source']]
                target = id_to_state[transition['target']]
                symbol = transition['symbol']
                input_symbols.update(symbol)

                if source not in transitions:
                    transitions[source] = {}

                # Manejar múltiples símbolos si es necesario
                if symbol not in transitions[source]:
                    transitions[source][symbol] = set()
                transitions[source][symbol].add(target)

            # Establecer el estado inicial
            if automata_data['initialState']:
                initial_state = id_to_state[automata_data['initialState']]

            # Establecer los estados finales
            for state_id in automata_data['acceptingStates']:
                state_name = id_to_state[state_id]
                final_states.add(state_name)

            # Crear instancias de State y configurar el NFA
            from .automata import State, NFA

            state_name_to_state = {}
            for state_name in states:
                is_initial = state_name == initial_state
                is_final = state_name in final_states
                state_instance = State(is_initial=is_initial, is_final=is_final)
                state_instance.name = state_name  # Asignar nombre al estado
                state_name_to_state[state_name] = state_instance

            # Configurar las transiciones
            for source_name, symbol_dict in transitions.items():
                source_state = state_name_to_state[source_name]
                for symbol, target_names in symbol_dict.items():
                    for target_name in target_names:
                        target_state = state_name_to_state[target_name]
                        source_state.add_transition(symbol, target_state)

            # Crear el NFA
            start_state = state_name_to_state[initial_state]
            accept_states = [state_name_to_state[state] for state in final_states]

            nfa = NFA(start_state, accept_states)

            # Convertir NFA a DFA
            dfa = nfa_to_dfa(nfa)

            # Generar gramática a partir del DFA
            initial_variable, N, T, P = dfa_to_grammar(dfa)

            # Determinar el tipo de gramática
            grammar_type = determine_grammar_type(N, T, P)

            # Agrupar producciones por variable
            productions_grouped = {}
            for variable, production in P:
                if variable not in productions_grouped:
                    productions_grouped[variable] = []
                productions_grouped[variable].append(production)

            language_description = generate_language_description(" | ".join(T))

            print(f"language_description: {language_description}")

            # Preparar el contexto para enviar al frontend
            context = {
                'initial_variable': initial_variable,
                'non_terminals': list(N),
                'terminals': list(T),
                'productions_grouped': productions_grouped,
                'grammar_type': grammar_type,
                'language_description': language_description
            }

            # Retornar el contexto como JSON
            return JsonResponse(context)
        except Exception as e:
            # Imprimir el traceback en la consola del servidor
            print("Error in draw_automata view:")
            traceback.print_exc()
            # Retornar una respuesta de error al cliente
            return JsonResponse({'error': str(e)}, status=500)
    else:
        # Manejar la solicitud GET
        return render(request, 'draw_automata.html')

def test_strings(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        test_strings = data.get('test_strings', [])
        automata_data = data.get('automata')
        state_id_map = data.get('stateIdMap')

        # Reconstruir el DFA a partir de los datos del autómata
        # Mapear los IDs a los nombres de los estados
        id_to_state = {state_id: state_name for state_id, state_name in state_id_map.items()}

        # Crear instancias de State
        states = {}
        start_state = None
        accept_states = []

        for state_id in automata_data['states']:
            state_name = id_to_state[state_id]
            is_initial = (state_id == automata_data['initialState'])
            is_final = (state_id in automata_data['acceptingStates'])
            state = State(is_initial=is_initial, is_final=is_final)
            state.name = state_name
            states[state_name] = state

            if is_initial:
                start_state = state
            if is_final:
                accept_states.append(state)

        # Configurar las transiciones
        for transition in automata_data['transitions']:
            source_name = id_to_state[transition['source']]
            target_name = id_to_state[transition['target']]
            symbol = transition['symbol']

            source_state = states[source_name]
            target_state = states[target_name]

            source_state.add_transition(symbol, target_state)

        # Crear el NFA
        nfa = NFA(start_state=start_state, accept_states=accept_states)

        # Convertir NFA a DFA
        dfa = nfa_to_dfa(nfa)

        # Simular las cadenas de prueba
        results = []
        for test_string in test_strings:
            is_accepted = simulate_dfa(dfa, test_string)
            results.append(is_accepted)

        # Retornar los resultados
        return JsonResponse({'results': results})
    else:
        return JsonResponse({'error': 'Invalid request method.'})

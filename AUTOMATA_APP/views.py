from datetime import datetime
from lib2to3.fixes.fix_input import context

from django.shortcuts import render
from .forms import RegexForm
from .automata import regex_to_nfa, remove_lambda_transitions, nfa_to_dfa, visualize_automata, reset_state_counter, regex_to_grammar, simulate_dfa
from .language_description import generate_language_description, determine_grammar_type

def index(request):
    current_year = datetime.now().year
    return render(request, 'index.html', {'current_year': current_year})

def regex_to_automata(request):
    if request.method == 'POST':
        form = RegexForm(request.POST)
        if form.is_valid():
            regex = form.cleaned_data['regex']
            reset_state_counter()
            nfa_lambda = regex_to_nfa(regex)
            reset_state_counter()
            nfa_no_lambda = remove_lambda_transitions(nfa_lambda)
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
    # Nueva vista para dibujar el autómata y mostrar información
    # Puedes replicar la lógica de 'regex_to_automata' o ajustarla según tus necesidades
    if request.method == 'POST':
        # Procesar los datos enviados por el usuario
        # ...
        context = {
            # Variables que se enviarán al template
            # ...
        }
        return render(request, 'draw_automata.html', context)
    else:
        # Mostrar el formulario inicial
        return render(request, 'draw_automata.html')

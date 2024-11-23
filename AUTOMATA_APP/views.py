from django.shortcuts import render
from .forms import RegexForm
from .automata import regex_to_nfa, remove_lambda_transitions, nfa_to_dfa, visualize_automata, reset_state_counter, regex_to_grammar, simulate_dfa
from .language_description import generate_language_description

def index(request):
    return render(request, 'regex_to_automata.html')

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
                ('AFN-λ', nfa_lambda_graph, 'nfa_lambda'),
                ('AFN', nfa_no_lambda_graph, 'nfa_no_lambda'),
                ('AFD', dfa_graph, 'dfa')
            ]

            # Generar la gramática utilizando el NFA existente
            initial_variable, N, T, P = regex_to_grammar(nfa_lambda)

            # Manejar cadenas de prueba
            test_strings = request.POST.get('test_strings', '')
            test_results = []
            if test_strings:
                test_strings_list = test_strings.strip().split('\n')
                for s in test_strings_list:
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

            context = {
                'form': form,
                'automata_list': automata_list,
                'initial_variable': initial_variable,
                'non_terminals': N,
                'terminals': T,
                'productions_grouped': productions_grouped,
                'language_description': language_description,
                'test_results': test_results
            }
            return render(request, 'regex_to_automata.html', context)
    else:
        form = RegexForm()
    return render(request, 'regex_to_automata.html', {'form': form})

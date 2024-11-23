from django import forms

class RegexForm(forms.Form):
    regex = forms.CharField(label='Expresión Regular', max_length=100)

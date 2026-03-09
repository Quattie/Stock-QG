from django import forms


class TickerForm(forms.Form):
    ticker = forms.CharField(
        label='',
        max_length=10,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. AAPL, TSLA, MSFT',
            'autocomplete': 'off',
        })
    )


class CryptoForm(forms.Form):
    crypto = forms.CharField(
        label='',
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. bitcoin, eth, solana',
            'autocomplete': 'off',
        })
    )


class EpochForm(forms.Form):
    epochs = forms.IntegerField(
        label='Training Epochs',
        max_value=100,
        min_value=1,
        initial=30,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
        })
    )

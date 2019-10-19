from django import forms


class TickerForm(forms.Form):
    ticker = forms.CharField(
        label='', max_length=8, widget=forms.TextInput(attrs={'class': 'tickerform'}))


class CryptoForm(forms.Form):
    crypto = forms.CharField(
        label='', max_length=8, widget=forms.TextInput(attrs={'class': 'cryptoform'})
    )


class EpochForm(forms.Form):
    epochs = forms.IntegerField(
        label="Amount of desired training epochs", max_value=50)

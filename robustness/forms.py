from django import forms


class UploadForm(forms.Form):
    model = forms.CharField()
    img = forms.FileField()


class ResponseForm(forms.Form):
    ori_path = forms.CharField()
    adv_path = forms.CharField()
    adversarial = forms.CharField()
    model = forms.CharField()

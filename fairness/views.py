from django.shortcuts import render, redirect
from django.http import HttpResponse
import json
from .fairness import fairness
from robustness.views import wrapper


def index(request):
    '''
    
    fairness:
    获取去偏评估页面
    
    '''

    return render(request, 'fairness/index.html', locals())


def start(request):
    '''
    
    showimg:
    返回去偏结果
    
    '''
    if request.method == 'POST':
        cont = {
            "raw_path": './static/fairness/fairness_raw.png',
            "demo_path": './static/fairness/fairness_aug1.png',
            "adv_path": './static/fairness/fairness_aug2.png',
        }
        fairness(cont["raw_path"], cont["demo_path"], cont["adv_path"])
        # return render(request, 'fairness/showimg.html', context=cont)
        cont = cont = {
            "raw_path": 'http://hzla.f3322.net:21200/static/fairness/fairness_raw.png',
            "demo_path": 'http://hzla.f3322.net:21200/static/fairness/fairness_aug1.png',
            "adv_path": 'http://hzla.f3322.net:21200/static/fairness/fairness_aug2.png',
        }
        return HttpResponse(json.dumps(cont, ensure_ascii=False))
    return redirect("/fairness")

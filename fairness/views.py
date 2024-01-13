from django.shortcuts import render, redirect
from django.http import HttpResponse
import json
from .fairness import fairness
from robustness.views import wrapper, HTTP_PREFIX, PORT


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
            "cor_path": {
                'pic1': './static/fairness/fairness_aug3.png',
                'pic2': './static/fairness/fairness_aug4.png',
                'pic3': './static/fairness/fairness_aug5.png',
            }
        }
        fairness(cont["raw_path"], cont["demo_path"], cont["adv_path"], cont["cor_path"])
        # return render(request, 'fairness/showimg.html', context=cont)
        cont = {
            "raw_path": f'{HTTP_PREFIX}:{PORT}/static/fairness/fairness_raw.png',
            "demo_path": f'{HTTP_PREFIX}:{PORT}/static/fairness/fairness_aug1.png',
            "adv_path": f'{HTTP_PREFIX}:{PORT}/static/fairness/fairness_aug2.png',
            "cor_path": {
                'pic1': f'{HTTP_PREFIX}:{PORT}/static/fairness/fairness_aug3.png',
                'pic2': f'{HTTP_PREFIX}:{PORT}/static/fairness/fairness_aug4.png',
                'pic3': f'{HTTP_PREFIX}:{PORT}/static/fairness/fairness_aug5.png',
            }
        }
        return HttpResponse(json.dumps(cont, ensure_ascii=False))
    return redirect("/fairness")

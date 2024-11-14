from django.shortcuts import render
from django.http import JsonResponse
from dotenv import load_dotenv
import os
import uuid
from nameko import config
from nameko.standalone.rpc import ClusterRpcProxy
from django.views.decorators.csrf import csrf_exempt



load_dotenv()
# Create your views here.
def home_page(request):
    return render(request, 'index.html')

@csrf_exempt
def chat_with_llama(request):
    session_id = str(uuid.uuid4())
    if request.method == "POST":
        user_message = request.POST.get('message')
        cloudamqp_uri = os.getenv('CLOUDAMQP_URI', 'default_api_key')

        config.setup({"AMQP_URI": cloudamqp_uri})
        with ClusterRpcProxy(context_data={"session_data": {}}) as rpc:
            res = rpc.bruna.send_help(session_id, user_message)

        return JsonResponse({"message": res.get("content")})
set -e

echo "Criando Banco de Dados..."
python3 manage.py makemigrations
python3 manage.py migrate



echo "Iniciando o Projeto..."


python3 manage.py runserver 0.0.0.0:8000

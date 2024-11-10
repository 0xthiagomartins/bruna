from .models import Usuario, Paciente, Supervisor, Especialista
from sqlmodel_controller import Controller
from . import engine


ctrl_usuario = Controller[Usuario](engine=engine.get())
ctrl_paciente = Controller[Paciente](engine=engine.get())
ctrl_supervisor = Controller[Supervisor](engine=engine.get())
ctrl_especialista = Controller[Especialista](engine=engine.get())

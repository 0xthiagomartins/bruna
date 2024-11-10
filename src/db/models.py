from sqlmodel import Field, JSON, Column, Relationship
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlmodel_controller import BaseID
from datetime import date
import enum


class Genero(str, enum.Enum):
    masculino = "masculino"
    feminino = "feminino"


class Usuario(BaseID, table=True):
    __tablename__ = "usuarios"

    email: str = Field(unique=True, index=True, nullable=False)
    nome: str = Field(nullable=True)
    contato: str = Field(nullable=True)
    senha: str = Field(nullable=False)
    verificado: bool = Field(default=True)
    genero: Genero = Field(sa_column=Column(SQLAlchemyEnum(Genero)))


class Paciente(BaseID, table=True):
    __tablename__ = "pacientes"

    usuario_id: int = Field(foreign_key="usuarios.id")


class Especialista(BaseID, table=True):
    __tablename__ = "especialistas"

    usuario_id: int = Field(foreign_key="usuarios.id")


class Supervisor(BaseID, table=True):
    __tablename__ = "supervisors"

    usuario_id: int = Field(foreign_key="usuarios.id")

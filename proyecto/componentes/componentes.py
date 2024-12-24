import dataclasses

@dataclasses.dataclass(frozen=True)
class Impacto:
    id_impacto: int
    x: float
    y: float
    z: float
    id_modulo: int
    id_rastro: int

    def __getitem__(self, indice):
        return (self.x, self.y, self.z)[indice]
    
    def __eq__(self, valor: object) -> bool:
        if self.id_impacto == valor.id_impacto:
            return True
        else:
            return False

@dataclasses.dataclass(frozen=True)
class Modulo:
    id_modulo: int
    z: float
    lx: float
    ly: float
    impactos: list[Impacto]
    
    def __eq__(self, valor: object) -> bool:
        if self.id_modulo == valor.id_modulo:
            return True
        else:
            return False

@dataclasses.dataclass
class InfoMC:
    vertice_primario: tuple
    theta: float
    phi: float

@dataclasses.dataclass(frozen=True)
class Rastro:
    id_rastro: int
    info_mc: InfoMC
    impactos: list[Impacto]
    
    def __eq__(self, valor: object) -> bool:
        if self.id_rastro == valor.id_rastro:
            return True
        else:
            return False

@dataclasses.dataclass(frozen=True)
class Evento:
    modulos: list[Modulo]
    rastros: list[Rastro]
    impactos: list[Impacto]

@dataclasses.dataclass(frozen=True)
class Segmento:
    id_segmento: int
    impacto_desde: Impacto
    impacto_hasta: Impacto
    
    def __eq__(self, valor: object) -> bool:
        if self.id_segmento == valor.id_segmento:
            return True
        else:
            return False
    
    def a_vector(self):
        return (self.impacto_hasta.x - self.impacto_desde.x, 
                self.impacto_hasta.y - self.impacto_desde.y, 
                self.impacto_hasta.z - self.impacto_desde.z)
    
    def __mul__(self, valor):
        vector_1 = self.a_vector()
        vector_2 = valor.a_vector()
        norma_1 = (vector_1[0]**2 + vector_1[1]**2 + vector_1[2]**2)**0.5
        norma_2 = (vector_2[0]**2 + vector_2[1]**2 + vector_2[2]**2)**0.5
        
        return (vector_1[0]*vector_2[0] + vector_1[1]*vector_2[1] + vector_1[2]*vector_2[2])/(norma_1*norma_2)

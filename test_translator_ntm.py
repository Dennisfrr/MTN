class ParallelNTMPrototype:
    """
    Simula não apenas a tradução (Spike Train), mas também o escoamento
    PARALELO E SIMULTÂNEO da tensão dentro do Grafo Topológico. 
    Mostra o que acontece quando a água de dois inputs colide fisicamente.
    """
    def __init__(self, time_ticks=40):
        self.time_ticks = time_ticks
        
        # O Transdutor Sensorial
        # Porta 0 recebe intensidade muito fraca (1)
        # Porta 1 recebe intensidade brutal (8)
        self.input_intensities = [2, 8] 
        self.accumulators = [0.0, 0.0]
        self.threshold = 1.0

        # O Grafo Físico (Topologia)
        # As tensões entram pela Porta 0 e Porta 1 e desaguam numa encruzilhada (Hub Central)
        self.tension_n0 = 0.0
        self.tension_n1 = 0.0
        self.hub_tension = 0.0 # Bacia acumulada do centro

        # Registro de História (Para o Terminal)
        self.history_p0 = []
        self.history_p1 = []
        self.history_hub = []
        
        # Estatísticas Globais
        self.colisoes_diretas = 0

    def run(self):
        print("=== SIMulacao NTM: TRADUCAO ESPACO-TEMPORAL E FLUXO PARALELO ===")
        print("1. As intensidades estaticas viram gotas eletricas temporais (Traducao).")
        print("2. A rede escoa as tensoes simultaneamente (Paralelismo Termodinamico).")
        print("-" * 65)

        for tick in range(self.time_ticks):
            # ====================================================
            # ETAPA 1: TRANSDUÇÃO SIMULTÂNEA NAS LENTES SENSORIAIS
            # ====================================================
            inject_port0 = 0.0
            inject_port1 = 0.0
            
            # Sensor da Porta 0
            self.accumulators[0] += self.input_intensities[0] / 10.0
            if self.accumulators[0] >= self.threshold:
                inject_port0 = 1.0  # Soltou uma Gota Físico/Tensão
                self.accumulators[0] -= self.threshold

            # Sensor da Porta 1
            self.accumulators[1] += self.input_intensities[1] / 10.0
            if self.accumulators[1] >= self.threshold:
                inject_port1 = 1.0 
                self.accumulators[1] -= self.threshold

            # As tensões recém-criadas brotam nas portas iniciais da topologia
            self.tension_n0 += inject_port0
            self.tension_n1 += inject_port1

            # ====================================================
            # ETAPA 2: GRAVIDADE NEURAL E FLUXO PARALELO
            # (Todos os nós movem a água escoando pro Hub ao mesmo tempo)
            # ====================================================
            
            # A agua despenca simultaneamente pro Hub
            flow_p0_to_hub = self.tension_n0  
            flow_p1_to_hub = self.tension_n1  
            
            # As margens esvaziam pq a agua fluiu
            self.tension_n0 = 0.0 
            self.tension_n1 = 0.0
            
            # Registro das gotas visuais no console (usando ASCII simples)
            self.history_p0.append("|" if flow_p0_to_hub > 0 else " ")
            self.history_p1.append("#" if flow_p1_to_hub > 0 else " ")
            
            # O Hub central (A Encrusilhada) recebe o impacto SIMULTÂNEO das duas portas
            # Perceba que a soma matemática só é possível porque elas viajam no mesmo 'tick'
            impacto_total_no_hub = flow_p0_to_hub + flow_p1_to_hub
            self.hub_tension += impacto_total_no_hub

            # Avaliando Colisão / Competição Física
            if flow_p0_to_hub > 0 and flow_p1_to_hub > 0:
                self.history_hub.append("X") # Colisão Brutal Simultânea
                self.colisoes_diretas += 1
            elif impacto_total_no_hub > 0:
                self.history_hub.append("v") # Tensão passando isoladamente
            else:
                self.history_hub.append("=") # Seca (Hub vazio e sem atividade)
                
            # No final do tick, o Hub próprio escoa lentamente adiante na rede
            # Simulando o alívio termodinâmico para as próximas camadas não modeladas aqui.
            self.hub_tension = max(0.0, self.hub_tension - 1.5)

        # Print Output
        print(f"Porta 0 (forca fraca) : [{''.join(self.history_p0)}]")
        print(f"Porta 1 (forca forte) : [{''.join(self.history_p1)}]")
        print(f"Hub de Decisao        : [{''.join(self.history_hub)}]")
        
        print("\n--- LEGENDA GEOMETRICA DO HUB CENTRAL ---")
        print("[ = ] -> Seco, nada passou por enquanto.")
        print("[ v ] -> Atravessou Tensao de apenas uma das fontes (sem atrito).")
        print("[ X ] -> COLISAO FISICA ESTRUTURAL.")
        print(f"         > As tensoes de duas fontes radicalmente diferentes colidiram {self.colisoes_diretas} vezes.")
        print("         > Nesses instantes, a Matematica Estatica diria 'somei dois numeros'.")
        print("         > A Fisica (NTM) diz: A pressao transbordou e vai forcar Neurogenese (Ramificacao Lateral).")

if __name__ == '__main__':
    sim = ParallelNTMPrototype(time_ticks=40)
    sim.run()

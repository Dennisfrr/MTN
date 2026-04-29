import numpy as np
from collections import defaultdict

class Edge:
    def __init__(self, dst, config, r_max=900.0, r_min=1.0, decay_rate=1.05):
        self.dst = dst
        self.config = config
        self.R_MAX = r_max
        self.R_MIN = r_min
        self.R = self.R_MAX
        self.DECAY_RATE = decay_rate
        self.total_flow_history = 0.0
        self.working_memory_pressure = 0.0 

    def conduct_allocated(self, allocated_flow_by_pid, read_only=False):
        total_flow = sum(allocated_flow_by_pid.values())
        self.working_memory_pressure += total_flow 
        if not read_only and total_flow > 1e-9:
            fatigue_relax = 1.0 + (total_flow * 0.5)
            self.R = max(self.R / fatigue_relax, self.R_MIN)
            self.total_flow_history += total_flow
            
    def decay(self):
        self.R = min(self.R * self.DECAY_RATE, self.R_MAX)


class SinkConnection:
    def __init__(self, r_max=900.0, r_min=1.0, decay_rate=1.05):
        self.R_MAX = r_max
        self.R_MIN = r_min
        self.R = self.R_MAX
        self.DECAY_RATE = decay_rate
        self.volume_collected = 0.0
        self.lifetime_flow = 0.0
        self.working_memory_pressure = 0.0 

    def conduct_allocated(self, allocated_flow_by_pid, read_only=False):
        total_flow = sum(allocated_flow_by_pid.values())
        self.working_memory_pressure += total_flow 
        if not read_only:
            self.volume_collected += total_flow
            self.lifetime_flow += total_flow
            if total_flow > 1e-9:
                fatigue_relax = 1.0 + (total_flow * 0.5)
                self.R = max(self.R / fatigue_relax, self.R_MIN)
        return total_flow

    def decay(self):
        self.R = min(self.R * self.DECAY_RATE, self.R_MAX)


class NTMGraph:
    def __init__(self, n_inputs, actions, config):
        self.n_inputs = n_inputs
        self.config = config
        
        self.edges = {}
        self.out_edges = defaultdict(list)
        self.node_tension = defaultdict(lambda: defaultdict(float))
        self.node_depth = {}
        self.next_node_id = n_inputs * 2
        self.edge_id_counter = 0

        # Ratos biológicos genéricos presentes no tecido inteiro
        self.node_sinks = defaultdict(lambda: {
            a: SinkConnection(self.config["R_MAX"], self.config["R_MIN"], self.config["DECAY_RATE"])
            for a in actions
        })

        # Inicializa Input -> Hidden layer 0 (Básico)
        for i in range(self.n_inputs):
            self.node_depth[i] = 0
            self._add_edge(i, self.n_inputs + i, depth=0)

    def _add_edge(self, src, dst, depth=0):
        eid = self.edge_id_counter
        self.edge_id_counter += 1
        self.edges[eid] = Edge(dst, self.config, self.config["R_MAX"], self.config["R_MIN"], self.config["DECAY_RATE"])
        self.out_edges[src].append(eid)
        if dst not in self.node_depth:
            self.node_depth[dst] = depth + 1
        return eid

    def _branch_node(self, src, active_nodes_pool):
        # Neurogênese Orgânica (Sem imposto Global ou trava arbitrária)
        candidate_targets = []
        # Tenta conectar a algum vizinho adjacente já ativo que não seja ele mesmo
        if active_nodes_pool:
            local_pool = [n for n in active_nodes_pool if n != src and n not in [self.edges[e].dst for e in self.out_edges[src]]]
            if local_pool:
                candidate_targets.extend(local_pool[:2])
                
        # Ou rasga o espaço criando um neurônio 100% virgem no tecido adjacente
        if not candidate_targets or np.random.rand() < 0.3:
            new_node = self.next_node_id
            self.next_node_id += 1
            candidate_targets.append(new_node)
            
        for dst in candidate_targets:
            self._add_edge(src, dst, depth=self.node_depth.get(src, 0))

    def propagate(self, input_vectors, target_action_by_pid=None, read_only=False):
        """
        Calcula pressão contínua e natural. Sem estrangulamentos matemáticos artificiais.
        A água bate e divide perfeitamente conforme a física de condutância manda.
        """
        for pid, vec in input_vectors.items():
            for i, t in enumerate(vec[:self.n_inputs]):
                if t > 0:
                    self.node_tension[i][pid] += t

        queue = list(self.node_tension.keys())
        visited = set()
        active_eids = set()
        active_sinks = set()
        
        global_leakage = defaultdict(float)

        while queue:
            nid = queue.pop(0)
            if nid in visited: continue
            visited.add(nid)
            
            t_dict = self.node_tension[nid]
            if sum(t_dict.values()) < 1e-9:
                continue
                
            # INI: LIMITE METABÓLICO DE SUPERFÍCIE (Synaptic Homeostasis - Orgânico)
            MEMBRANE_LIMIT = 2.5 # Suporta cerca de 2 fluxos max maciços + algumas dezenas de capilares
            membrane_used = sum(1.0 / self.edges[e].R for e in self.out_edges[nid])
            membrane_used += sum(1.0 / snk.R for snk in self.node_sinks[nid].values())
            membrane_free = max(0.0, MEMBRANE_LIMIT - membrane_used)
            # FIM: LIMITE DE SUPERFÍCIE

            # ============== SINK CFD (Fluido Contínuo Orgânico) =================
            comp_by_p = defaultdict(float)
            for p in t_dict:
                for a_name, snk in self.node_sinks[nid].items():
                    if target_action_by_pid is None or target_action_by_pid.get(p) == a_name:
                        choke_f = 1.0 / (1.0 + snk.working_memory_pressure * 0.5)
                        comp_by_p[p] += (1.0 / snk.R) * choke_f

            sink_allocated = defaultdict(lambda: defaultdict(float))
            sink_delivered = defaultdict(lambda: defaultdict(float))
            for p, t_rem in t_dict.items():
                if t_rem < 1e-9: continue
                total_desired = 0.0
                desired_map = {}
                for a_name, snk in self.node_sinks[nid].items():
                    if target_action_by_pid is None or target_action_by_pid.get(p) == a_name:
                        choke_f = 1.0 / (1.0 + snk.working_memory_pressure * 0.5)
                        c = (1.0 / snk.R) * choke_f
                        fraction = c / comp_by_p[p] if comp_by_p[p] > 0 else 0.0
                        fraction *= np.random.uniform(0.98, 1.02)
                        num_comp = comp_by_p[p] / c if c > 0 else 1.0
                        excl = 1.0 / num_comp if num_comp > 0 else 0.0
                        
                        eps = 1e-6
                        # Ganho Biológico Honesto (Tensão vs Inércia)
                        desired = (t_rem * (1.0 + t_rem)) * fraction
                        desired_map[a_name] = desired
                        total_desired += desired
                        
                for a_name, desired in desired_map.items():
                    snk = self.node_sinks[nid][a_name]
                    c = 1.0 / snk.R
                    limit = 15.0 * c * (1.0 + excl)
                    
                    water = min(desired, limit)
                    if water > eps:
                        sink_allocated[a_name][p] += water
                        sink_delivered[a_name][p] += water

            for a_name, snk in self.node_sinks[nid].items():
                if sink_allocated[a_name]:
                    old_R = snk.R
                    pulled = snk.conduct_allocated(sink_delivered[a_name], read_only)
                    
                    if not read_only and snk.R < old_R:
                        c_increase = (1.0 / snk.R) - (1.0 / old_R)
                        if c_increase <= membrane_free:
                            membrane_free -= c_increase
                        else:
                            max_c = (1.0 / old_R) + membrane_free
                            if max_c > 0: snk.R = max(snk.R, 1.0 / max_c)
                            membrane_free = 0.0
                            
                    global_leakage[a_name] += pulled
                    for p, val in sink_allocated[a_name].items():
                        self.node_tension[nid][p] = max(0.0, self.node_tension[nid][p] - val)
                    if pulled > 1e-9:
                        active_sinks.add((nid, a_name))

            # ============== EDGE CFD (Fluido Contínuo Orgânico) =================
            eids = self.out_edges[nid]
            if eids:
                edge_allocated = defaultdict(lambda: defaultdict(float))
                edge_delivered = defaultdict(lambda: defaultdict(float))
                
                total_edge_conductance = 0.0
                edge_chokes = {}
                for eid in eids:
                    edge = self.edges[eid]
                    choke = 1.0 / (1.0 + edge.working_memory_pressure * 0.5)
                    edge_chokes[eid] = choke
                    total_edge_conductance += (1.0 / edge.R) * choke
                
                for p, t_rem in t_dict.items():
                    if t_rem < 1e-9: continue
                    total_desired = 0.0
                    desired_map = {}
                    for eid in eids:
                        c = (1.0 / self.edges[eid].R) * edge_chokes[eid]
                        portion = c / total_edge_conductance if total_edge_conductance > 0 else 0.0
                        portion *= np.random.uniform(0.98, 1.02) 
                        
                        eps = 1e-6
                        desired = (t_rem * (1.0 + t_rem)) * portion
                        desired_map[eid] = desired
                        total_desired += desired
                        
                    for eid in desired_map:
                        if total_desired > t_rem:
                            actual_flow = t_rem * (desired_map[eid] / total_desired)
                        else:
                            actual_flow = desired_map[eid]
                            
                        # Custo Térmico de Fluxo Extremo
                        loss = (actual_flow ** 1.5) * 0.05
                        delivered = max(0.0, actual_flow - loss)
                        limit = 15.0 * (1.0 / self.edges[eid].R) * edge_chokes[eid]
                        actual_flow = min(actual_flow, limit)
                        
                        if actual_flow > 1e-6:
                            edge_allocated[eid][p] += actual_flow
                            edge_delivered[eid][p] += delivered

                for eid in eids:
                    if edge_allocated[eid]:
                        old_R = self.edges[eid].R
                        self.edges[eid].conduct_allocated(edge_delivered[eid], read_only)
                        
                        if not read_only and self.edges[eid].R < old_R:
                            c_increase = (1.0 / self.edges[eid].R) - (1.0 / old_R)
                            if c_increase <= membrane_free:
                                membrane_free -= c_increase
                            else:
                                max_c = (1.0 / old_R) + membrane_free
                                if max_c > 0: self.edges[eid].R = max(self.edges[eid].R, 1.0 / max_c)
                                membrane_free = 0.0
                                
                        active_eids.add(eid)
                        for p, val in edge_allocated[eid].items():
                            self.node_tension[nid][p] = max(0.0, self.node_tension[nid][p] - val)
                            self.node_tension[self.edges[eid].dst][p] += edge_delivered[eid][p]
                        if self.edges[eid].dst not in visited and self.edges[eid].dst not in queue:
                            queue.append(self.edges[eid].dst)

            # ============== RUPTURA ESTOCÁSTICA =================
            if not read_only:
                total_tension_left = sum(self.node_tension[nid].values())
                # Sem limites arbitrários ridículos. Apenas probabilidade termodinâmica pura baseada na membrana natural.
                base_c = 1.0 / self.config["R_MAX"]
                if membrane_free >= base_c and len(self.out_edges[nid]) < 20: 
                    CONSTANTE_THERMICA = 40.0
                    prob = 1.0 - np.exp(-total_tension_left / CONSTANTE_THERMICA)
                    if np.random.rand() < prob:
                        self._branch_node(nid, active_nodes_pool=queue)
                        membrane_free -= base_c

        if not read_only:
            for eid, edge in self.edges.items():
                if eid not in active_eids:
                    edge.decay()
            for nid_visited in visited:
                if nid_visited in self.node_sinks:
                    for a_name, s in self.node_sinks[nid_visited].items():
                        if (nid_visited, a_name) not in active_sinks:
                            s.decay()
                            
        # Escoamento temporal do Back-pressure Biológico (Água saindo dos canos abertos)
        for edge in self.edges.values():
            edge.working_memory_pressure *= 0.5 
        for s_dict in self.node_sinks.values():
            for snk in s_dict.values():
                snk.working_memory_pressure *= 0.5

        # Perda Térmica (Dissipação Contínua)
        LEAK = 0.01
        for ds_nid in self.node_tension:
            for p, val in list(self.node_tension[ds_nid].items()):
                if val > 1e-9:
                    self.node_tension[ds_nid][p] = val * (1.0 - LEAK)
                else:
                    self.node_tension[ds_nid][p] = 0.0

        return global_leakage

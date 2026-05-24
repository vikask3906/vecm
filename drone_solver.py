import json
import sys
import math
import heapq

def sum_path_dist(path_steps, start_node, nodes):
    d = 0.0
    curr = start_node
    for v, arr, dep in path_steps:
        d += math.hypot(nodes[curr][0]-nodes[v][0], nodes[curr][1]-nodes[v][1])
        curr = v
    return d

def get_spatial_intersection(u, v, D, dx, dy, nfz):
    if D == 0: return None, None
    if nfz.get('shape') == 'circle':
        cx, cy = nfz.get('center', [0,0])
        r = nfz.get('radius', 0) + 1e-4
        ux_c = u[0] - cx
        uy_c = u[1] - cy
        b = ux_c * dx + uy_c * dy
        c = ux_c * ux_c + uy_c * uy_c - r * r
        disc = b * b - c
        if disc < 0:
            return None, None
        sq = math.sqrt(max(0.0, disc))
        s1 = -b - sq
        s2 = -b + sq
        d_in = max(0.0, s1)
        d_out = min(D, s2)
        if d_in < d_out:
            return d_in, d_out
        return None, None
    else:
        corners = nfz.get('corners', [])
        if not corners: return None, None
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        xmin = min(xs) - 1e-4
        xmax = max(xs) + 1e-4
        ymin = min(ys) - 1e-4
        ymax = max(ys) + 1e-4
        
        if abs(dx) > 1e-9:
            tx1 = (xmin - u[0]) / dx
            tx2 = (xmax - u[0]) / dx
            tx_in = min(tx1, tx2)
            tx_out = max(tx1, tx2)
        else:
            if xmin <= u[0] <= xmax:
                tx_in = -float('inf')
                tx_out = float('inf')
            else:
                return None, None
                
        if abs(dy) > 1e-9:
            ty1 = (ymin - u[1]) / dy
            ty2 = (ymax - u[1]) / dy
            ty_in = min(ty1, ty2)
            ty_out = max(ty1, ty2)
        else:
            if ymin <= u[1] <= ymax:
                ty_in = -float('inf')
                ty_out = float('inf')
            else:
                return None, None
                
        s_in = max(0.0, tx_in, ty_in)
        s_out = min(D, tx_out, ty_out)
        if s_in < s_out:
            return s_in, s_out
        return None, None

def is_point_in_nfz(u, nfz):
    if nfz.get('shape') == 'circle':
        cx, cy = nfz.get('center', [0,0])
        r = nfz.get('radius', 0) + 1e-4
        return math.hypot(u[0] - cx, u[1] - cy) <= r
    else:
        corners = nfz.get('corners', [])
        if not corners: return False
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        xmin = min(xs) - 1e-4
        xmax = max(xs) + 1e-4
        ymin = min(ys) - 1e-4
        ymax = max(ys) + 1e-4
        return xmin <= u[0] <= xmax and ymin <= u[1] <= ymax

def precompute_graph(warehouse, deliveries, charging_stations, nfzs, map_size):
    nodes = []
    nodes.append(tuple(warehouse))
    WH_IDX = 0
    
    deliv_idxs = {}
    for d in deliveries:
        pos = (d.get('x',0), d.get('y',0))
        if pos not in nodes:
            nodes.append(pos)
        deliv_idxs[d.get('id')] = nodes.index(pos)
        
    cs_idxs = {}
    for i, cs in enumerate(charging_stations):
        pos = (cs.get('x',0), cs.get('y',0))
        if pos not in nodes:
            nodes.append(pos)
        cs_idxs[i] = nodes.index(pos)
        
    eps = 1e-2
    for nfz in nfzs:
        if nfz.get('shape') == 'circle':
            cx, cy = nfz.get('center', [0,0])
            r = nfz.get('radius', 0) + eps
            for i in range(16):  
                theta = i * math.pi / 8
                nx, ny = cx + r * math.cos(theta), cy + r * math.sin(theta)
                if 0 <= nx <= map_size[0] and 0 <= ny <= map_size[1]:
                    if (nx, ny) not in nodes:
                        nodes.append((nx, ny))
        else:
            corners = nfz.get('corners', [])
            if corners:
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)
                for nx, ny in [(xmin - eps, ymin - eps), (xmax + eps, ymin - eps),
                               (xmin - eps, ymax + eps), (xmax + eps, ymax + eps)]:
                    if 0 <= nx <= map_size[0] and 0 <= ny <= map_size[1]:
                        if (nx, ny) not in nodes:
                            nodes.append((nx, ny))
                            
    num_nodes = len(nodes)
    point_in_nfz = [[False]*len(nfzs) for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        for k, nfz in enumerate(nfzs):
            point_in_nfz[i][k] = is_point_in_nfz(nodes[i], nfz)
            
    return nodes, WH_IDX, deliv_idxs, cs_idxs, point_in_nfz

def get_edge_collisions(u, v, nodes, D, nfzs, edges_cache):
    edge_key = (u, v)
    if edge_key in edges_cache:
        return edges_cache[edge_key]
    
    res = []
    if D > 0:
        U = nodes[u]
        V = nodes[v]
        dx, dy = (V[0]-U[0])/D, (V[1]-U[1])/D
        for k, nfz in enumerate(nfzs):
            d_in, d_out = get_spatial_intersection(U, V, D, dx, dy, nfz)
            if d_in is not None and d_in < d_out:
                res.append((k, d_in, d_out))
    edges_cache[edge_key] = res
    return res

def plan_path(A_idx, t_A, B_idx, nodes, nfzs, point_in_nfz, edges_cache):
    if A_idx == B_idx:
        return [], t_A
    num_nodes = len(nodes)
    pq = []
    dist_AB = math.hypot(nodes[A_idx][0]-nodes[B_idx][0], nodes[A_idx][1]-nodes[B_idx][1])
    counter = 0
    heapq.heappush(pq, (t_A + dist_AB, t_A, counter, A_idx))
    earliest = {A_idx: t_A}
    parent = {} 
    
    while pq:
        f, t, _, u = heapq.heappop(pq)
        
        if t > earliest.get(u, float('inf')):
            continue
            
        if u == B_idx:
            path_steps = []
            curr = B_idx
            while curr != A_idx:
                p, t_arr, t_dep = parent[curr]
                path_steps.append((curr, t_arr, t_dep))
                curr = p
            path_steps.reverse()
            return path_steps, t
            
        for v in range(num_nodes):
            if u == v: continue
            
            dist_uv = math.hypot(nodes[u][0]-nodes[v][0], nodes[u][1]-nodes[v][1])
            t_dep = t
            valid = True
            while True:
                max_t = t_dep
                collisions = get_edge_collisions(u, v, nodes, dist_uv, nfzs, edges_cache)
                for nfz_idx, d_in, d_out in collisions:
                    nfz = nfzs[nfz_idx]
                    T_start = nfz.get('T_start', 0)
                    T_end = nfz.get('T_end', float('inf'))
                    if t_dep + d_in < T_end + 1e-4 and t_dep + d_out > T_start - 1e-4:
                        t_needed = T_end - d_in + 1e-3
                        if t_needed > max_t:
                            max_t = t_needed
                if max_t == t_dep:
                    break
                t_dep = max_t
                
            if t_dep > t:
                for k, nfz in enumerate(nfzs):
                    if point_in_nfz[u][k]:
                        T_start = nfzs[k].get('T_start', 0)
                        T_end = nfzs[k].get('T_end', float('inf'))
                        if t < T_end + 1e-4 and t_dep > T_start - 1e-4:
                            valid = False
                            break
            
            if not valid:
                continue
                
            t_arr = t_dep + dist_uv
            if t_arr < earliest.get(v, float('inf')):
                earliest[v] = t_arr
                parent[v] = (u, t_arr, t_dep)
                dist_vB = math.hypot(nodes[v][0]-nodes[B_idx][0], nodes[v][1]-nodes[B_idx][1])
                counter += 1
                heapq.heappush(pq, (t_arr + dist_vB, t_arr, counter, v))
                
    return None

def plan_macro_leg(start_node, t_start, energy_start, dest_node, payload, nodes, cs_idxs, temp_slots, nfzs, point_in_nfz, edges_cache):
    if start_node == dest_node:
        return t_start, energy_start, [], temp_slots
        
    pq = []
    h_start = math.hypot(nodes[start_node][0]-nodes[dest_node][0], nodes[start_node][1]-nodes[dest_node][1])
    counter = 0
    heapq.heappush(pq, (t_start + h_start, t_start, counter, start_node, energy_start, [], temp_slots))
    
    best_t_for_node = {}
    
    while pq:
        f, t_curr, _, u, e_curr, history, curr_slots = heapq.heappop(pq)
        
        if len(history) > 15:
            continue
            
        state_key = u
        if state_key in best_t_for_node:
            prev_t, prev_e = best_t_for_node[state_key]
            if prev_t <= t_curr and prev_e >= e_curr:
                continue
        best_t_for_node[state_key] = (t_curr, e_curr)
        
        if u == dest_node:
            return t_curr, e_curr, history, curr_slots
            
        neighbors = [dest_node] + list(cs_idxs.values())
        neighbors = list(set(neighbors)) 
        neighbors.sort(key=lambda v: math.hypot(nodes[u][0]-nodes[v][0], nodes[u][1]-nodes[v][1]))
        
        selected_neighbors = []
        for v in neighbors:
            if len(selected_neighbors) >= 8 and v != dest_node:
                continue
            selected_neighbors.append(v)
            
        for v in selected_neighbors:
            if u == v: continue
            
            dist_uv_straight = math.hypot(nodes[u][0]-nodes[v][0], nodes[u][1]-nodes[v][1])
            if e_curr < dist_uv_straight * (1.0 + payload) - 1e-4:
                continue 
                
            res = plan_path(u, t_curr, v, nodes, nfzs, point_in_nfz, edges_cache)
            if not res: continue
            p_steps, t_arr_v = res
            
            dist_uv_exact = sum_path_dist(p_steps, u, nodes)
            e_used = dist_uv_exact * (1.0 + payload)
            
            if e_curr < e_used - 1e-4:
                continue 
                
            e_arr_v = e_curr - e_used
            
            if v == dest_node:
                new_history = history + [(u, v, p_steps, t_curr, t_arr_v, None)]
                counter += 1
                heapq.heappush(pq, (t_arr_v, t_arr_v, counter, v, e_arr_v, new_history, curr_slots))
            else:
                cs_raw = None
                for k, val in cs_idxs.items():
                    if val == v:
                        cs_raw = k
                        break
                if cs_raw is None: continue
                
                slots = curr_slots[cs_raw]
                if not slots: continue 
                
                best_slot_idx = min(range(len(slots)), key=lambda i: slots[i])
                t_start_charge = max(t_arr_v, slots[best_slot_idx])
                duration = max(0.0, (500.0 - e_arr_v) / 2.0)
                t_dep_v = t_start_charge + duration
                
                valid_wait = True
                for k, nfz in enumerate(nfzs):
                    if point_in_nfz[v][k]:
                        T_start = nfz.get('T_start', 0)
                        T_end = nfz.get('T_end', float('inf'))
                        if t_arr_v < T_end + 1e-4 and t_dep_v > T_start - 1e-4:
                            valid_wait = False
                            break
                if not valid_wait:
                    continue
                
                new_slots = [list(s) for s in curr_slots]
                new_slots[cs_raw][best_slot_idx] = t_dep_v
                
                charge_info = {
                    't_start_charge': t_start_charge,
                    't_dep': t_dep_v,
                    'slot_idx': best_slot_idx,
                    'cs_raw': cs_raw
                }
                
                new_history = history + [(u, v, p_steps, t_curr, t_arr_v, charge_info)]
                h_v = math.hypot(nodes[v][0]-nodes[dest_node][0], nodes[v][1]-nodes[dest_node][1])
                counter += 1
                heapq.heappush(pq, (t_dep_v + h_v, t_dep_v, counter, v, 500.0, new_history, new_slots))
                
    return None

def try_batch(drone, batch, t_start, cs_slots_state, nodes, WH_IDX, deliv_idxs, cs_idxs, point_in_nfz, nfzs, map_size, edges_cache):
    total_weight = sum(d.get('weight',0) for d in batch)
    max_pay = drone.get('max_payload', float('inf'))
    if total_weight > max_pay + 1e-6: return None
    
    curr_t = t_start
    curr_node = WH_IDX
    curr_energy = 500.0
    curr_payload = total_weight
    
    actions = []
    actions.append({
        "x": round(nodes[WH_IDX][0], 2), "y": round(nodes[WH_IDX][1], 2),
        "t": round(curr_t, 2), "action": "PICKUP",
        "delivery_ids": [d.get('id') for d in batch]
    })
    
    curr_slots = [list(s) for s in cs_slots_state]
    
    for d in batch:
        dest_idx = deliv_idxs[d.get('id')]
        
        res = plan_macro_leg(curr_node, curr_t, curr_energy, dest_idx, curr_payload, nodes, cs_idxs, curr_slots, nfzs, point_in_nfz, edges_cache)
        if not res: return None
        
        t_arr, e_arr, history, new_slots = res
        deadline = d.get('deadline', float('inf'))
        if t_arr > deadline: return None
        
        for u, v, p_steps, t_dep_u, t_arr_v, c_info in history:
            c_node = u
            c_t = t_dep_u
            for wp, arr_wp, dep_wp in p_steps:
                if dep_wp > c_t + 1e-3:
                    actions.append({"x": round(nodes[c_node][0], 2), "y": round(nodes[c_node][1], 2), "t": round(dep_wp, 2), "action": "WAIT"})
                if wp != v:
                    actions.append({"x": round(nodes[wp][0], 2), "y": round(nodes[wp][1], 2), "t": round(arr_wp, 2), "action": "WAYPOINT"})
                c_node = wp
                c_t = arr_wp
                
            if c_info: 
                cs_idx = v
                t_start_charge = c_info['t_start_charge']
                t_dep_cs = c_info['t_dep']
                if t_start_charge > t_arr_v + 1e-3:
                    actions.append({"x": round(nodes[cs_idx][0], 2), "y": round(nodes[cs_idx][1], 2), "t": round(t_start_charge, 2), "action": "WAIT"})
                actions.append({"x": round(nodes[cs_idx][0], 2), "y": round(nodes[cs_idx][1], 2), "t": round(t_start_charge, 2), "action": "CHARGE"})
                actions.append({"x": round(nodes[cs_idx][0], 2), "y": round(nodes[cs_idx][1], 2), "t": round(t_dep_cs, 2), "action": "CHARGE_COMPLETE"})
                
        curr_payload -= d.get('weight', 0)
        actions.append({
            "x": round(nodes[dest_idx][0], 2), "y": round(nodes[dest_idx][1], 2),
            "t": round(t_arr, 2), "action": "DELIVER", "delivery_id": d.get('id')
        })
        curr_node = dest_idx
        curr_t = t_arr
        curr_energy = e_arr
        curr_slots = new_slots

    res_wh = plan_macro_leg(curr_node, curr_t, curr_energy, WH_IDX, 0.0, nodes, cs_idxs, curr_slots, nfzs, point_in_nfz, edges_cache)
    if not res_wh: return None
    
    t_arr, e_arr, history, new_slots = res_wh
    
    for u, v, p_steps, t_dep_u, t_arr_v, c_info in history:
        c_node = u
        c_t = t_dep_u
        for wp, arr_wp, dep_wp in p_steps:
            if dep_wp > c_t + 1e-3:
                actions.append({"x": round(nodes[c_node][0], 2), "y": round(nodes[c_node][1], 2), "t": round(dep_wp, 2), "action": "WAIT"})
            if wp != v:
                actions.append({"x": round(nodes[wp][0], 2), "y": round(nodes[wp][1], 2), "t": round(arr_wp, 2), "action": "WAYPOINT"})
            c_node = wp
            c_t = arr_wp
            
        if c_info: 
            cs_idx = v
            t_start_charge = c_info['t_start_charge']
            t_dep_cs = c_info['t_dep']
            if t_start_charge > t_arr_v + 1e-3:
                actions.append({"x": round(nodes[cs_idx][0], 2), "y": round(nodes[cs_idx][1], 2), "t": round(t_start_charge, 2), "action": "WAIT"})
            actions.append({"x": round(nodes[cs_idx][0], 2), "y": round(nodes[cs_idx][1], 2), "t": round(t_start_charge, 2), "action": "CHARGE"})
            actions.append({"x": round(nodes[cs_idx][0], 2), "y": round(nodes[cs_idx][1], 2), "t": round(t_dep_cs, 2), "action": "CHARGE_COMPLETE"})
            
    actions.append({"x": round(nodes[WH_IDX][0], 2), "y": round(nodes[WH_IDX][1], 2), "t": round(t_arr, 2), "action": "RETURN"})
    
    return actions, t_arr, new_slots

def find_best_batch(drone, t_start, unassigned, cs_slots_state, nodes, WH_IDX, deliv_idxs, cs_idxs, point_in_nfz, nfzs, map_size, edges_cache):
    for i, d0 in enumerate(unassigned): 
        batch = [d0]
        res = try_batch(drone, batch, t_start, cs_slots_state, nodes, WH_IDX, deliv_idxs, cs_idxs, point_in_nfz, nfzs, map_size, edges_cache)
        if not res:
            continue
            
        best_res = res
        best_batch = list(batch)
        curr_weight = d0.get('weight', 0)
        curr_pos = (d0.get('x',0), d0.get('y',0))
        max_pay = drone.get('max_payload', float('inf'))
        
        candidates = [d for d in unassigned if d.get('id') not in [b.get('id') for b in batch]]
        while candidates:
            candidates.sort(key=lambda d: math.hypot(d.get('x',0)-curr_pos[0], d.get('y',0)-curr_pos[1]))
            added = False
            for cand in candidates[:3]:
                if curr_weight + cand.get('weight',0) <= max_pay + 1e-6:
                    test_batch = best_batch + [cand]
                    res_test = try_batch(drone, test_batch, t_start, cs_slots_state, nodes, WH_IDX, deliv_idxs, cs_idxs, point_in_nfz, nfzs, map_size, edges_cache)
                    if res_test:
                        best_res = res_test
                        best_batch = test_batch
                        curr_weight += cand.get('weight',0)
                        curr_pos = (cand.get('x',0), cand.get('y',0))
                        added = True
                        candidates.remove(cand)
                        break
            if not added:
                break
                
        return best_batch, best_res[0], best_res[1], best_res[2]
        
    return None

def solve(warehouse, drones, deliveries, no_fly_zones, charging_stations, map_size):
    nodes, WH_IDX, deliv_idxs, cs_idxs, point_in_nfz = precompute_graph(warehouse, deliveries, charging_stations, no_fly_zones, map_size)
    edges_cache = {}
    
    unassigned = sorted(deliveries, key=lambda d: d.get('deadline', float('inf')))
    drone_states = {d.get('id'): {'t_free': 0.0, 'path': []} for d in drones}
    cs_slots_state = [[0.0] * max(1, cs.get('slots', 1)) for cs in charging_stations if cs.get('slots', 1) > 0]
    
    while unassigned:
        any_assigned = False
        for did in sorted(drone_states.keys(), key=lambda x: drone_states[x]['t_free']):
            d_obj = next(d for d in drones if d.get('id') == did)
            t_start = drone_states[did]['t_free']
            
            b_info = find_best_batch(d_obj, t_start, unassigned, cs_slots_state, nodes, WH_IDX, deliv_idxs, cs_idxs, point_in_nfz, no_fly_zones, map_size, edges_cache)
            if b_info:
                batch, actions, t_return, new_slots = b_info
                drone_states[did]['path'].extend(actions)
                drone_states[did]['t_free'] = t_return
                cs_slots_state = new_slots
                for d in batch:
                    unassigned.remove(d)
                any_assigned = True
                break
                
        if not any_assigned:
            break
            
    flight_manifest = []
    for did, state in drone_states.items():
        if state['path']:
            flight_manifest.append({
                "drone_id": did,
                "path": state['path']
            })
            
    return flight_manifest

if __name__ == '__main__':
    try:
        input_data_str = sys.stdin.read().strip()
        if not input_data_str:
            sys.exit(0)
        input_data = json.loads(input_data_str)
        map_size = input_data.get('map_size', [10000, 10000])
        warehouse = [map_size[0] / 2.0, map_size[1] / 2.0]
        drones = input_data.get('drones') or []
        deliveries = input_data.get('deliveries') or []
        no_fly_zones = input_data.get('no_fly_zones') or []
        charging_stations = input_data.get('charging_stations') or []
        
        result = solve(warehouse, drones, deliveries, no_fly_zones, charging_stations, map_size)
        output = {"flight_manifest": result}
        print(json.dumps(output, indent=2))
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)

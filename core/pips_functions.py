import pandas as pd
import numpy as np
import math
from typing import Tuple, List
from models import TradeResult

def find_pips(data: np.ndarray, n_pips: int, dist_measure: int = 2) -> Tuple[List[int], List[float]]:
    """Extract perceptually important points from price data."""
    if len(data) < n_pips:
        return list(range(len(data))), data.tolist()
    
    close_data = data['close'].values
    pips_x = [0, len(close_data) - 1]  # Index
    pips_y = [close_data[0], close_data[-1]]  # Price
    
    for curr_point in range(2, n_pips):
        md = 0.0  # Max distance
        md_i = -1  # Max distance index
        insert_index = -1
        
        for k in range(0, curr_point - 1):
            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1
            
            if pips_x[right_adj] - pips_x[left_adj] <= 1:
                continue
            
            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope
            
            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                d = 0.0  # Distance
                
                if dist_measure == 1:  # Euclidean distance
                    d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - close_data[i]) ** 2) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - close_data[i]) ** 2) ** 0.5
                elif dist_measure == 2:  # Perpendicular distance
                    d = abs((slope * i + intercept) - close_data[i]) / (slope ** 2 + 1) ** 0.5
                else:  # Vertical distance
                    d = abs((slope * i + intercept) - close_data[i])
                
                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj
        
        if md_i != -1:
            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, close_data[md_i])
    
    high_data = data['high'].values
    low_data = data['low'].values
    
    for i, (pip_index, pip_price) in enumerate(zip(pips_x, pips_y)):
        if i > 0:
            prev_price = pips_y[i - 1]
            trend = pip_price - prev_price
            
            if trend >= 0:
                pips_y[i] = high_data[pip_index]
            else:
                pips_y[i] = low_data[pip_index]
    
    return pips_x, pips_y

def calculate_angle_at_middle_point(x1, y1, x2, y2, x3, y3):
    vector1 = np.array([x1 - x2, y1 - y2])  # P2 -> P1
    vector2 = np.array([x3 - x2, y3 - y2])  # P2 -> P3
    
    mag1 = np.linalg.norm(vector1)
    mag2 = np.linalg.norm(vector2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    dot_product = np.dot(vector1, vector2)
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_radians = math.acos(cos_angle)
    
    return (math.degrees(angle_radians))

def is_valid_standard_pattern(py, px, window_data) -> Tuple[bool, str]:

    if len(px) < 6:
        return False, ""

    rear = len(px) - 1
    pt1 = px[rear]
    pt2 = px[rear - 1]
    pt3 = px[rear - 2]
    pt4 = px[rear - 3]
    pt5 = px[rear - 4]
    pt6 = px[rear - 5]
    
    y1 = py[rear]
    y2 = py[rear - 1]
    y3 = py[rear - 2]
    y4 = py[rear - 3]
    y5 = py[rear - 4]
    y6 = py[rear - 5]
    slope_1_5 = (pt5 - pt1) / (y5 - y1)
    angle_radians_1_5 = math.atan(slope_1_5)


    angle2 = calculate_angle_at_middle_point(y1, pt1, y2, pt2, y3, pt3)
    angle4 = calculate_angle_at_middle_point(y3, pt3, y4, pt4, y5, pt5)
    angle5 = calculate_angle_at_middle_point(y4, pt4, y5, pt5, y6, pt6)

    angle3 = calculate_angle_at_middle_point(y1, pt1, y3, pt3, y5, pt5)

    if(abs(math.degrees(angle_radians_1_5)) < 20 and max(angle2, angle4, angle5) < 90 and angle3 < 100):
        if (pt3 > max(pt1, pt5) and pt2 < min(pt3, pt1) and pt4 < min(pt5, pt3)): #Bear
            return True, "Sell"
        if (pt3 < min(pt1, pt5) and pt2 > max(pt1, pt3) and pt4 > max(pt5, pt3)): #Bull
            return True, "Buy"

    return False, "Hold"

def merge_near_collinear_pips(px: List[int], py: List[float], angle_thresh_deg=3):
    k = 1
    while k < len(px) - 1:
        angle = calculate_angle_at_middle_point(px[k-1], py[k-1],
                                               px[k], py[k],
                                               px[k+1], py[k+1])
        
        if (angle > angle_thresh_deg):
            # remove the middle pivot
            px.pop(k)
            py.pop(k)  # do not increment k → re-test new triple
        else:
            k += 1
    
    return px, py

def create_stop_loss_profit(py, px, window_data, action, max_lot, max_loss):
    
    rear = len(px) - 1
    pt1 = px[rear]
    pt2 = px[rear - 1]
    pt3 = px[rear - 2]
    pt4 = px[rear - 3]
    pt5 = px[rear - 4]
    pt6 = px[rear - 5]
    
    y1 = py[rear]
    y2 = py[rear - 1]
    y3 = py[rear - 2]
    y4 = py[rear - 3]
    y5 = py[rear - 4]
    y6 = py[rear - 5]
    
    take_profit_pos = []
    
    can_trade = True
    points4_6 = window_data[y6 + 1 : y4]
    points1_2 = window_data[y2 : y1]

    if action == 'Buy':
        stop_loss = min(points4_6['low'].values.min(), points1_2['low'].values.min()) 
        take_profit_pos.append(max(pt2, pt4)) # first profit line
        diff = abs(take_profit_pos[0] - stop_loss)# adam theory
        take_profit_pos.append(take_profit_pos[0] + diff)
        can_trade = stop_loss > pt3
    elif action == 'Sell':
        stop_loss = max(points4_6['high'].values.max(), points1_2['high'].values.max())
        take_profit_pos.append(min(pt2, pt4))
        diff = abs(take_profit_pos[0] - stop_loss)# adam theory
        take_profit_pos.append(take_profit_pos[0] - diff)
        can_trade = stop_loss < pt3
    else:
        can_trade = False
        stop_loss = 0.0
    
    if stop_loss * max_lot > max_loss:
        max_lot = 1
    return stop_loss, take_profit_pos, can_trade, max_lot

def assign_trade_result(price, diff_points, exit_reason, point_value, trade_result, commission_fee):
    trade_result.exit_price = price
    trade_result.exit_reason = exit_reason
    trade_result.pnl_points += diff_points
    trade_result.pnl_dollars += (diff_points * point_value)
    trade_result.pnl_dollars -= commission_fee
    return trade_result

def analyze(data_frame, start_idx, stop_loss, take_profit_pos, action, initial_size, point_value, round_turn):
    
    # Initialize trade result
    entry_time = data_frame.index[start_idx]
    entry_price = (data_frame.iloc[start_idx - 1]['close']  + data_frame.iloc[start_idx - 1]['high']) * 0.5
    trade_result = TradeResult(
        entry_time=entry_time,
        entry_price=entry_price,
        action=action,
        pnl_dollars = -round_turn
    )
    if (abs(stop_loss - entry_price) * point_value > 800): 
        return start_idx + 1, trade_result
    
    current_size = 1
    # Track from next bar onwards
    for i in range(start_idx + 1, len(data_frame)):
        current_bar = data_frame.iloc[i]
        current_time = data_frame.index[i]
        
        if action == 'Buy':
            if current_bar['low'] <= stop_loss:
                if trade_result.exit_reason == None:
                    assign_trade_result(stop_loss, (stop_loss - entry_price) * (initial_size * 0.5), 'STOP_LOSS', point_value, trade_result, round_turn)
                else:
                    reason = 'TAKE_PROFIT + STOP_LOSS'
                    assign_trade_result(stop_loss, (stop_loss - entry_price)* (initial_size * 0.5), reason, point_value, trade_result, round_turn)
                return i, trade_result

            
            if current_bar['high'] >= take_profit_pos[0] or current_bar['high'] >= take_profit_pos[1]:
                if (current_bar['high'] >= take_profit_pos[1]):
                    if initial_size > 1:
                        current_size = int(initial_size * 0.5)
                    assign_trade_result(take_profit_pos[1], (take_profit_pos[1] - entry_price) * current_size, 'TAKE_PROFIT_2', point_value, trade_result, round_turn)
                    if (current_size >= initial_size):
                        return i, trade_result
                stop_loss = max(current_bar['low'], stop_loss)  # adjust stop loss

            if (trade_result.exit_reason != None):
                stop_loss = max(current_bar['low'], stop_loss)
        
        else:  # 'Sell'
            # Check stop loss (price goes above stop)
            if current_bar['high'] >= stop_loss:
                if trade_result.exit_reason == None:
                    reason = 'STOP_LOSS'
                    assign_trade_result(stop_loss, (entry_price - stop_loss) * initial_size, reason, point_value, trade_result, round_turn)
                else:
                    reason = 'TAKE_PROFIT_1 + STOP_LOSS'
                    assign_trade_result(stop_loss, (entry_price - stop_loss) * (initial_size * 0.5), reason, point_value, trade_result, round_turn)
                return i, trade_result
            
            if current_bar['low'] <= take_profit_pos[0] or current_bar['low'] <= take_profit_pos[1]:
                if (current_bar['low'] <= take_profit_pos[1]):
                    if initial_size > 1:
                        current_size = int(initial_size * 0.5)
                    assign_trade_result(take_profit_pos[1], (entry_price - take_profit_pos[1]) * current_size, 'TAKE_PROFIT_2', point_value, trade_result, round_turn)
                    if (current_size >= initial_size):
                        return i, trade_result
                    
                stop_loss = min(current_bar['high'], stop_loss)

            if (trade_result.exit_reason != None):
                stop_loss = min(current_bar['high'], stop_loss)

    
    # If we reach end of data without hitting stops or targets
    final_price = data_frame.iloc[-1]['close']
    trade_result.exit_time = data_frame.index[-1]
    trade_result.exit_price = final_price
    trade_result.exit_reason = 'END_OF_DATA'
    trade_result.duration_bars = len(data_frame) - 1 - start_idx
    
    if action == 'Buy':
        trade_result.pnl_points = final_price - entry_price
    else:
        trade_result.pnl_points = entry_price - final_price
    
    trade_result.pnl_dollars = trade_result.pnl_points * point_value
    
    return len(data_frame), trade_result

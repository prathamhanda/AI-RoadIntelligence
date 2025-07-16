#!/usr/bin/env python3
"""
🚦 Lane-Specific Traffic Control Logic Demonstration
Test the traffic control algorithm without requiring SUMO installation

Author: Pratham Handa
"""

# Simulate the traffic control parameters
LEFT_LANE_THRESHOLD = 5
RIGHT_LANE_THRESHOLD = 5
DENSITY_IMBALANCE_RATIO = 2.0

TRAFFIC_STATES = {
    'NORMAL_FLOW': "GGrGG",      # Both lanes green
    'LEFT_PRIORITY': "GGrrr",    # Prioritize left lane 
    'RIGHT_PRIORITY': "rrGGG",   # Prioritize right lane
    'EMERGENCY_STOP': "rrrrr",   # Emergency stop all traffic
    'ANIMAL_CAUTION': "yyyyy"    # Yellow caution for animal presence
}

def test_traffic_control_logic(vehicles_left, vehicles_right, violence_active=False, animal_active=False):
    """Test the lane-specific traffic control logic"""
    
    # Simulate global state variables
    global violence_detected_start, animal_detected_start
    violence_detected_start = 1 if violence_active else None
    animal_detected_start = 1 if animal_active else None
    
    # Analyze lane-specific traffic conditions
    left_heavy = vehicles_left >= LEFT_LANE_THRESHOLD
    right_heavy = vehicles_right >= RIGHT_LANE_THRESHOLD
    total_vehicles = vehicles_left + vehicles_right
    
    # Calculate density imbalance ratio
    if vehicles_right > 0:
        density_ratio = vehicles_left / vehicles_right
    else:
        density_ratio = float('inf') if vehicles_left > 0 else 1.0
        
    # Determine traffic control action based on lane analysis
    if violence_detected_start is not None:
        # Emergency: Stop all traffic for violence
        current_state = TRAFFIC_STATES['EMERGENCY_STOP']
        action_reason = f"🚨 EMERGENCY STOP - Violence detected"
        action_type = "EMERGENCY_VIOLENCE"
        
    elif animal_detected_start is not None:
        # Caution: Yellow lights for animal presence  
        current_state = TRAFFIC_STATES['ANIMAL_CAUTION']
        action_reason = f"🐕 ANIMAL CAUTION - {total_vehicles} vehicles detected"
        action_type = "ANIMAL_CAUTION"
        
    elif left_heavy and right_heavy:
        # Both lanes congested - normal flow to prevent deadlock
        current_state = TRAFFIC_STATES['NORMAL_FLOW']
        action_reason = f"🚦 BOTH LANES HEAVY (L:{vehicles_left}, R:{vehicles_right}) - Normal flow"
        action_type = "BOTH_LANES_HEAVY"
        
    elif left_heavy and not right_heavy:
        # Left lane congested - prioritize left lane
        current_state = TRAFFIC_STATES['LEFT_PRIORITY']
        action_reason = f"⬅️ LEFT LANE PRIORITY ({vehicles_left} vehicles) - Right clear ({vehicles_right})"
        action_type = "LEFT_PRIORITY"
        
    elif right_heavy and not left_heavy:
        # Right lane congested - prioritize right lane
        current_state = TRAFFIC_STATES['RIGHT_PRIORITY']
        action_reason = f"➡️ RIGHT LANE PRIORITY ({vehicles_right} vehicles) - Left clear ({vehicles_left})"
        action_type = "RIGHT_PRIORITY"
        
    elif density_ratio >= DENSITY_IMBALANCE_RATIO:
        # Significant left lane imbalance
        current_state = TRAFFIC_STATES['LEFT_PRIORITY']
        action_reason = f"⬅️ LEFT DENSITY IMBALANCE (ratio: {density_ratio:.1f}) - L:{vehicles_left}, R:{vehicles_right}"
        action_type = "DENSITY_IMBALANCE_LEFT"
        
    elif density_ratio <= (1.0 / DENSITY_IMBALANCE_RATIO):
        # Significant right lane imbalance  
        current_state = TRAFFIC_STATES['RIGHT_PRIORITY']
        action_reason = f"➡️ RIGHT DENSITY IMBALANCE (ratio: {density_ratio:.1f}) - L:{vehicles_left}, R:{vehicles_right}"
        action_type = "DENSITY_IMBALANCE_RIGHT"
        
    else:
        # Normal traffic conditions
        current_state = TRAFFIC_STATES['NORMAL_FLOW']
        action_reason = f"✅ NORMAL FLOW - Balanced traffic (L:{vehicles_left}, R:{vehicles_right})"
        action_type = "NORMAL_FLOW"
    
    return {
        "action": action_type,
        "reason": action_reason,
        "vehicles_left": vehicles_left,
        "vehicles_right": vehicles_right, 
        "traffic_state": current_state,
        "density_ratio": density_ratio
    }

def demo_traffic_scenarios():
    """Demonstrate various traffic scenarios"""
    print("🚦 Lane-Specific Traffic Control Logic Demonstration")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Left Lane Threshold: {LEFT_LANE_THRESHOLD} vehicles")
    print(f"  Right Lane Threshold: {RIGHT_LANE_THRESHOLD} vehicles") 
    print(f"  Density Imbalance Ratio: {DENSITY_IMBALANCE_RATIO}")
    print("=" * 70)
    
    # Test scenarios
    scenarios = [
        # (left_vehicles, right_vehicles, violence, animal, description)
        (3, 2, False, False, "Normal light traffic"),
        (6, 2, False, False, "Left lane congestion"),
        (2, 7, False, False, "Right lane congestion"),
        (8, 9, False, False, "Both lanes heavy"),
        (10, 3, False, False, "Left density imbalance"),
        (2, 8, False, False, "Right density imbalance"),
        (5, 4, True, False, "Violence emergency"),
        (6, 3, False, True, "Animal on road"),
        (0, 0, False, False, "No traffic"),
        (15, 1, False, False, "Extreme left imbalance"),
    ]
    
    for i, (left, right, violence, animal, desc) in enumerate(scenarios, 1):
        print(f"\n🔍 Scenario {i}: {desc}")
        print(f"   Input: Left={left}, Right={right}, Violence={violence}, Animal={animal}")
        
        result = test_traffic_control_logic(left, right, violence, animal)
        
        print(f"   Action: {result['action']}")
        print(f"   Traffic State: {result['traffic_state']}")
        print(f"   Density Ratio: {result['density_ratio']:.2f}")
        print(f"   Decision: {result['reason']}")

if __name__ == "__main__":
    demo_traffic_scenarios()

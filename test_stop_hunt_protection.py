"""
Test Stop Hunt Protection
Validates that SL avoids psychological levels and liq zones
"""

from smart_entry import SmartEntryAnalyzer

def test_stop_hunt_protection():
    print("=" * 70)
    print("Testing Stop Hunt Protection")
    print("=" * 70)
    
    analyzer = SmartEntryAnalyzer()
    
    # Test Case 1: SL near $100k (major level)
    print("\n🧪 Test 1: SL near $100k (major psychological level)")
    print("-" * 70)
    initial_sl = 99500  # Very close to 100k
    adjusted_sl, justification = analyzer.calculate_safe_sl(
        initial_sl=initial_sl,
        direction="LONG",
        entry_price=102000,
        liq_zones=None,
        current_price=102000
    )
    print(f"Initial SL: ${initial_sl:,.0f}")
    print(f"Adjusted SL: ${adjusted_sl:,.0f}")
    print(f"Justification: {justification}")
    assert adjusted_sl < 99500, "SL should move AWAY from $100k"
    print("✅ PASS: SL moved away from $100k")
    
    # Test Case 2: SL near $105k (intermediate level)
    print("\n🧪 Test 2: SL near $105k (intermediate level)")
    print("-" * 70)
    initial_sl = 105100
    adjusted_sl, justification = analyzer.calculate_safe_sl(
        initial_sl=initial_sl,
        direction="LONG",
        entry_price=107000,
        liq_zones=None,
        current_price=107000
    )
    print(f"Initial SL: ${initial_sl:,.0f}")
    print(f"Adjusted SL: ${adjusted_sl:,.0f}")
    print(f"Justification: {justification}")
    assert adjusted_sl < 105100, "SL should move away from $105k"
    print("✅ PASS: SL adjusted for intermediate level")
    
    # Test Case 3: SL safe (no nearby psych levels)
    print("\n🧪 Test 3: SL safe at $98,300 (no nearby psych levels)")
    print("-" * 70)
    initial_sl = 98300  # Far from 98k, 99k, 100k
    adjusted_sl, justification = analyzer.calculate_safe_sl(
        initial_sl=initial_sl,
        direction="LONG",
        entry_price=101000,
        liq_zones=None,
        current_price=101000
    )
    print(f"Initial SL: ${initial_sl:,.0f}")
    print(f"Adjusted SL: ${adjusted_sl:,.0f}")
    print(f"Justification: {justification}")
    assert adjusted_sl == 98300, f"SL should remain unchanged, got {adjusted_sl}"
    print("✅ PASS: SL kept unchanged (safe)")
    
    # Test Case 4: Psychological level detection
    print("\n🧪 Test 4: Psychological level detection")
    print("-" * 70)
    levels = analyzer._detect_psychological_levels(price=105000, scan_range_pct=5.0)  # Wider range
    print(f"Detected {len(levels)} levels near $105,000 (±5%):")
    for level in levels[:8]:
        print(f"  • ${level['level']:,.0f} ({level['type']}) - {level['distance_pct']:.2f}% away")
    assert any(l['level'] == 105000 and l['type'] == 'intermediate' for l in levels), "Should detect $105k intermediate"
    assert any(l['level'] == 100000 and l['type'] == 'major' for l in levels), "Should detect $100k major"
    assert any(l['level'] == 110000 and l['type'] == 'major' for l in levels), "Should detect $110k major"
    # Note: minors ($1k) removed from detection
    print("✅ PASS: Psychological levels detected correctly")
    
    # Test Case 5: SHORT direction (SL above entry)
    print("\n🧪 Test 5: SHORT direction near $110k")
    print("-" * 70)
    initial_sl = 110200  # Just above 110k
    adjusted_sl, justification = analyzer.calculate_safe_sl(
        initial_sl=initial_sl,
        direction="SHORT",
        entry_price=108000,
        liq_zones=None,
        current_price=108000
    )
    print(f"Initial SL: ${initial_sl:,.0f}")
    print(f"Adjusted SL: ${adjusted_sl:,.0f}")
    print(f"Justification: {justification}")
    assert adjusted_sl > 110200, "SL should move ABOVE $110k for SHORT"
    print("✅ PASS: SHORT SL moved correctly")
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    test_stop_hunt_protection()

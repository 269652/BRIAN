#!/usr/bin/env python3
"""Diagnostic: Why are reg_* metrics not appearing in training logs?

Run this in your Colab to debug the missing OOD intervention telemetry.
"""
import sys
import torch

def diagnose_regularization():
    print("=" * 70)
    print("PR2 OOD INTERVENTION DIAGNOSTIC")
    print("=" * 70)
    
    # 1. Check if regularizers module exists
    print("\n[1] Checking neuroslm.regularizers module...")
    try:
        from neuroslm import regularizers
        print("✓ neuroslm.regularizers imported successfully")
        print(f"  Module path: {regularizers.__file__}")
        
        # Check for RegularizationController
        if hasattr(regularizers, "RegularizationController"):
            print("✓ RegularizationController class found")
        else:
            print("✗ RegularizationController class NOT FOUND")
            print("  Available classes:", dir(regularizers))
            return False
    except ImportError as e:
        print(f"✗ Failed to import neuroslm.regularizers: {e}")
        return False
    
    # 2. Check DSL regularization config
    print("\n[2] Checking DSL regularization config...")
    try:
        from neuroslm.dsl.regularization import RegularizationConfig
        print("✓ RegularizationConfig imported")
        
        cfg = RegularizationConfig()
        print(f"  Default dar.enabled: {cfg.dar.enabled}")
        print(f"  Default pcc.enabled: {cfg.pcc.enabled}")
        print(f"  Default isotropy.enabled: {cfg.isotropy.enabled}")
        print(f"  Default cmd.enabled: {cfg.cmd.enabled}")
        print(f"  any_enabled(): {cfg.any_enabled()}")
    except Exception as e:
        print(f"✗ Failed to load RegularizationConfig: {e}")
        return False
    
    # 3. Load arch and check regularization block
    print("\n[3] Checking arch.neuro regularization block...")
    try:
        from neuroslm.dsl.parser import parse_architecture_file
        
        arch_path = "architectures/rcc_bowtie/arch.neuro"
        print(f"  Parsing: {arch_path}")
        
        ast = parse_architecture_file(arch_path)
        
        # Find regularization block
        reg_block = None
        for node in ast:
            if hasattr(node, "name") and node.name == "regularization":
                reg_block = node
                break
        
        if reg_block:
            print("✓ Found regularization {} block in arch.neuro")
            # Try to extract fields
            if hasattr(reg_block, "fields"):
                for field in reg_block.fields:
                    print(f"    {field.name}: ...")
        else:
            print("✗ No regularization {} block found in arch.neuro")
            return False
            
    except Exception as e:
        print(f"✗ Failed to parse arch.neuro: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Build a harness and check reg_controller
    print("\n[4] Building BRIANHarness and checking reg_controller...")
    try:
        from neuroslm.dsl.nn_lang import build_language_model
        from neuroslm.harness import BRIANHarness
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.dsl.regularization import RegularizationConfig, DARConfig, PCCConfig
        
        # Create minimal training config with regularization
        reg_cfg = RegularizationConfig(
            dar=DARConfig(enabled=True, weight=0.1),
            pcc=PCCConfig(enabled=True, weight=0.5),
        )
        
        train_cfg = TrainingConfig(
            loss_clip=True,
            regularization=reg_cfg,
        )
        
        # Build tiny model
        lm = build_language_model(
            vocab=1000,
            d_model=128,
            depth=2,
            n_heads=4,
            max_ctx=512,
        )
        
        harness = BRIANHarness(
            language_model=lm,
            training_config=train_cfg,
        )
        
        print("✓ BRIANHarness created")
        
        if hasattr(harness, "reg_controller"):
            print(f"✓ harness.reg_controller exists: {harness.reg_controller}")
            
            # Check which interventions are enabled
            if hasattr(harness.reg_controller, "dar"):
                print(f"  DAR module: {harness.reg_controller.dar}")
            if hasattr(harness.reg_controller, "pcc"):
                print(f"  PCC module: {harness.reg_controller.pcc}")
        else:
            print("✗ harness.reg_controller NOT FOUND")
            return False
        
        # 5. Try a forward pass
        print("\n[5] Testing forward pass with reg metrics...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        harness = harness.to(device)
        
        ids = torch.randint(0, 1000, (2, 32), device=device)
        targets = torch.randint(0, 1000, (2, 32), device=device)
        
        # Forward
        loss = harness.compute_loss(ids, targets)
        print(f"✓ compute_loss() succeeded: loss={loss.item():.4f}")
        
        # Check metrics
        print("\n[6] Checking published metrics...")
        if hasattr(harness, "_metrics"):
            metrics = harness._metrics
            print(f"  Total metrics: {len(metrics)}")
            
            reg_keys = [k for k in metrics.keys() if k.startswith("reg_")]
            if reg_keys:
                print(f"✓ Found {len(reg_keys)} reg_* metrics:")
                for k in reg_keys:
                    print(f"    {k}: {metrics[k]:.6f}")
            else:
                print("✗ NO reg_* metrics found!")
                print(f"  Available metrics: {list(metrics.keys())}")
                return False
        else:
            print("✗ harness._metrics doesn't exist")
            return False
            
    except Exception as e:
        print(f"✗ Failed during harness test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL CHECKS PASSED — OOD interventions should be working!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = diagnose_regularization()
    sys.exit(0 if success else 1)

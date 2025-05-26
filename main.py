# main.py

import argparse
import json
import os
import logging


# Simple logging setup
def setup_logger(level="INFO", log_file=None):
    """Simple logger setup"""
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger('tft_trading_bot')


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_config(config):
    """Simple configuration validation"""
    required_sections = ['data', 'model', 'strategy', 'execution']
    for section in required_sections:
        if section not in config:
            print(f"Missing required section: {section}")
            return False
    return True


def load_model(model_path):
    """Load a trained model"""
    import torch
    from models.tft.model import TemporalFusionTransformer, SimpleTFT

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract model config from checkpoint if available
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        # Use default config with enhanced settings
        model_config = {
            'hidden_size': 64,
            'attention_heads': 4,
            'lstm_layers': 2,
            'dropout': 0.1,
            'past_sequence_length': 120,
            'forecast_horizon': 12,
            'quantiles': [0.1, 0.5, 0.9],
            'static_input_dim': 1,
            'past_input_dim': 50,
            'future_input_dim': 10
        }

    # Try to create TFT model first, fallback to SimpleTFT if needed
    try:
        # Check if the checkpoint contains TFT-specific layers
        state_dict_keys = checkpoint.get('model_state_dict', {}).keys()
        has_tft_layers = any('attention' in key or 'variable_selection' in key for key in state_dict_keys)

        if has_tft_layers:
            # Create full TFT model
            model = TemporalFusionTransformer(model_config)
        else:
            # Use SimpleTFT for simpler checkpoints
            model = SimpleTFT(model_config)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            # Only load weights that exist in the model
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

        model.eval()
        return model

    except Exception as e:
        # If TFT fails, fallback to SimpleTFT
        print(f"Failed to load as TFT model: {e}")
        print("Falling back to SimpleTFT...")

        model = SimpleTFT(model_config)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

        model.eval()
        return model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced TFT Trading Bot")
    parser.add_argument('--config', type=str, default='config/config.json',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'live', 'export'],
                        default='train', help='Operation mode')
    parser.add_argument('--model-type', type=str, choices=['full', 'simple'],
                        default='full', help='Model type to use (full TFT or simplified)')
    args = parser.parse_args()

    # Load and validate configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print("Make sure config.json exists with your OANDA credentials")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return

    if not validate_config(config):
        print("❌ Invalid configuration. Exiting.")
        return

    # Setup logging
    log_config = config.get('logging', {})
    logger = setup_logger(
        log_config.get('level', 'INFO'),
        log_config.get('file', 'logs/trading_bot.log')
    )
    logger.info(f"Starting Enhanced TFT Trading Bot in {args.mode} mode")

    if args.mode == 'train':
        try:
            # Import modules
            from data.collectors.oanda_collector import OandaDataCollector
            from data.processors.normalizer import DataNormalizer
            from models.tft.model import TemporalFusionTransformer  # Use full TFT
            from pipelines.training.trainer import TFTTrainer
            from data.dataset import create_datasets

            # Collect and process data
            logger.info("Collecting data from OANDA...")
            collector = OandaDataCollector(config)
            data = collector.collect_training_data()

            # Process data
            logger.info("Processing and normalizing data...")
            processor = DataNormalizer(config)
            processed_data = processor.process(data)

            # Create dataset and loaders
            logger.info("Creating datasets...")
            train_loader, val_loader, test_loader = create_datasets(processed_data, config)

            # Initialize full TFT model
            logger.info("Initializing Enhanced Temporal Fusion Transformer...")
            model = TemporalFusionTransformer(config['model'])

            # Initialize model with a dummy batch to create all layers
            logger.info("Initializing model layers with sample data...")
            dummy_batch = next(iter(train_loader))

            # Move model to appropriate device first
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Move dummy batch to same device
            for key in dummy_batch:
                if isinstance(dummy_batch[key], torch.Tensor):
                    dummy_batch[key] = dummy_batch[key].to(device)

            # Run a forward pass to initialize all layers
            with torch.no_grad():
                try:
                    _ = model(dummy_batch)
                    logger.info("Model layers successfully initialized")
                except Exception as e:
                    logger.error(f"Error during model initialization: {e}")
                    # Fallback to SimpleTFT if TFT fails
                    logger.info("Falling back to SimpleTFT model...")
                    from models.tft.model import SimpleTFT
                    model = SimpleTFT(config['model']).to(device)

            # Print model information after initialization
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")

            if total_params == 0:
                logger.error("Model has no parameters! Falling back to SimpleTFT...")
                from models.tft.model import SimpleTFT
                model = SimpleTFT(config['model']).to(device)
                # Re-check parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(
                    f"SimpleTFT model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")

            # Train model
            logger.info(f"Starting training with {model.__class__.__name__}...")
            trainer = TFTTrainer(config, model, train_loader, val_loader, test_loader)

            training_config = config.get('training', {})
            if training_config.get('use_cross_validation', False):
                test_metrics = trainer.train_with_cross_validation(
                    n_splits=training_config.get('cross_validation_folds', 5)
                )
            else:
                test_metrics = trainer.train()

            logger.info(f"Training completed. Test metrics: {test_metrics}")

            # Export model if configured
            export_config = config.get('export', {})
            if export_config.get('auto_export_after_training', False):
                try:
                    from pipelines.export.onnx_exporter import ONNXExporter
                    export_path = os.path.join(export_config.get('export_dir', 'exported_models'),
                                               'enhanced_tft_model.onnx')
                    exporter = ONNXExporter(config)

                    # Create a dummy input for ONNX export
                    dummy_batch = next(iter(train_loader))
                    success = exporter.export_model(model, export_path, dummy_batch)

                    if success:
                        logger.info(f"Enhanced TFT model exported to {export_path}")
                    else:
                        logger.error("Failed to export Enhanced TFT model")
                except Exception as e:
                    logger.error(f"Error during model export: {e}")

        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.error("Please ensure all required modules are available")
            return
        except Exception as e:
            logger.error(f"Training error: {e}")
            return

    elif args.mode == 'backtest':
        try:
            from strategy.strategy_factory import create_strategy
            from execution.execution_engine import ExecutionEngine

            # Load trained model
            backtest_config = config.get('backtest', {})
            model_path = backtest_config.get('model_path', 'models/checkpoints/best_model.pt')

            logger.info("Loading Enhanced TFT model for backtesting...")
            model = load_model(model_path)

            # Create enhanced strategy
            strategy = create_strategy(config, strategy_type='enhanced_tft')

            # Create execution engine with enhanced components
            engine = ExecutionEngine(config, model, strategy)

            # Run backtest with enhanced analytics
            logger.info("Starting enhanced backtest...")
            results = engine.backtest()

            # Print/save results with enhanced metrics
            logger.info(f"Enhanced backtest completed. Results: {results}")

            # Generate detailed performance report
            if results.get('success', False):
                try:
                    from performance_analyzer import PerformanceAnalyzer

                    analyzer = PerformanceAnalyzer(
                        trades=results.get('trades', []),
                        equity_curve=results.get('equity_curve', []),
                        config=config
                    )

                    # Generate comprehensive report
                    report = analyzer.generate_performance_report(
                        output_file=f"backtest_results/enhanced_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    )

                    logger.info("Enhanced performance report generated")

                except Exception as e:
                    logger.warning(f"Could not generate enhanced performance report: {e}")

        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return

    elif args.mode == 'live':
        try:
            from strategy.strategy_factory import create_strategy
            from execution.execution_engine import ExecutionEngine

            # Load trained model
            live_config = config.get('live', {})
            model_path = live_config.get('model_path', 'models/checkpoints/best_model.pt')

            logger.info("Loading Enhanced TFT model for live trading...")
            model = load_model(model_path)

            # Create enhanced strategy with quality controls
            strategy = create_strategy(config, strategy_type='enhanced_tft')

            # Create execution engine with advanced features
            engine = ExecutionEngine(config, model, strategy)

            # Start live trading with enhanced monitoring
            logger.info("Starting enhanced live trading...")
            engine.start_live_trading()

        except Exception as e:
            logger.error(f"Live trading error: {e}")
            return

    elif args.mode == 'export':
        try:
            from pipelines.export.onnx_exporter import ONNXExporter

            # Load trained model
            export_config = config.get('export', {})
            model_path = export_config.get('model_path', 'models/checkpoints/best_model.pt')

            logger.info("Loading Enhanced TFT model for export...")
            model = load_model(model_path)

            # Export enhanced model
            export_path = os.path.join(export_config.get('export_dir', 'exported_models'), 'enhanced_tft_model.onnx')
            exporter = ONNXExporter(config)

            # Need a dummy batch for export
            logger.info("Loading sample data for Enhanced TFT ONNX export...")
            from data.collectors.oanda_collector import OandaDataCollector
            from data.processors.normalizer import DataNormalizer
            from data.dataset import create_datasets

            collector = OandaDataCollector(config)
            data = collector.collect_training_data()
            processor = DataNormalizer(config)
            processed_data = processor.process(data)
            train_loader, _, _ = create_datasets(processed_data, config)
            dummy_batch = next(iter(train_loader))

            success = exporter.export_model(model, export_path, dummy_batch)
            if success:
                logger.info(f"Enhanced TFT model exported to {export_path}")
            else:
                logger.error("Failed to export Enhanced TFT model")

        except Exception as e:
            logger.error(f"Export error: {e}")
            return


if __name__ == "__main__":
    main()
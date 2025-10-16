import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from phenotypic.analysis._log_growth_model import LogGrowthModel, LOG_GROWTH_MODEL


class TestLogGrowthModel:
    """Test suite for LogGrowthModel functionality."""

    @pytest.fixture(scope="class")
    def measurement_data(self):
        """Load measurement data for testing."""
        data_path = Path(__file__).parent.parent/"src"/"phenotypic"/"data"/"meas"/"meas.csv"
        df = pd.read_csv(data_path, nrows=10000)  # Load first 10k nrows for testing
        return df

    @pytest.fixture(scope="class")
    def expected_results(self):
        """Load expected fitted results for validation."""
        data_path = Path(__file__).parent.parent/"src"/"phenotypic"/"data"/"meas"/"growth-rates-area.csv"
        df = pd.read_csv(data_path)
        return df

    @pytest.fixture
    def sample_data(self):
        """Create small synthetic dataset for unit testing."""
        np.random.seed(42)

        # Create synthetic logistic growth data
        time_points = np.arange(0, 10, 1)
        r_true, K_true, N0_true = 0.5, 1000, 50

        # Generate logistic growth with noise
        t_data = []
        size_data = []

        for t in time_points:
            size = K_true/(1 + (K_true - N0_true)/N0_true*np.exp(-r_true*t))
            size_noisy = size + np.random.normal(0, size*0.05)  # 5% noise
            t_data.extend([t]*3)  # 3 replicates per time point
            size_data.extend([size_noisy]*3)

        df = pd.DataFrame({
            'Metadata_Time'     : t_data,
            'Shape_Area'        : size_data,
            'Metadata_Dataset'  : ['Test']*len(t_data),
            'Metadata_Strain'   : ['Strain1']*len(t_data),
            'Metadata_Replicate': list(range(len(t_data)))
        })

        return df

    def test_initialization(self):
        """Test LogGrowthModel initialization with various parameters."""
        # Test basic initialization
        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain']
        )

        assert model.on == 'Shape_Area'
        assert model.groupby == ['Metadata_Dataset', 'Metadata_Strain']
        assert model.time_label == 'Metadata_Time'
        assert model.Kmax_label is None
        assert model.reg_factor == 1.2
        assert model.kmax_penalty == 5
        assert model.loss == "linear"
        assert not model.verbose
        assert model.n_jobs == 1

        # Test custom parameters
        model_custom = LogGrowthModel(
                on='Area',
                groupby=['Dataset', 'Strain'],
                time_label='Time',
                Kmax_label='Max_Capacity',
                growth_rate_penalty=2.0,
                cap_penalty=10,
                loss="linear",
                verbose=True,
                n_jobs=2
        )

        assert model_custom.on == 'Area'
        assert model_custom.groupby == ['Dataset', 'Strain']
        assert model_custom.time_label == 'Time'
        assert model_custom.Kmax_label == 'Max_Capacity'
        assert model_custom.reg_factor == 2.0
        assert model_custom.kmax_penalty == 10
        assert model_custom.verbose
        assert model_custom.n_jobs == 2

    def test_analyze_basic(self, sample_data):
        """Test basic model analysis functionality."""
        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain'],
                verbose=False
        )

        results = model.analyze(sample_data)

        # Check that results DataFrame has expected structure
        assert isinstance(results, pd.DataFrame)
        assert not results.empty

        # Check that all expected columns are present
        expected_columns = [
            LOG_GROWTH_MODEL.R_FIT,
            LOG_GROWTH_MODEL.K_FIT,
            LOG_GROWTH_MODEL.N0_FIT,
            LOG_GROWTH_MODEL.GROWTH_RATE,
            LOG_GROWTH_MODEL.K_MAX,
            LOG_GROWTH_MODEL.NUM_SAMPLES,
            LOG_GROWTH_MODEL.LOSS,
            LOG_GROWTH_MODEL.STATUS,
            LOG_GROWTH_MODEL.MAE,
            LOG_GROWTH_MODEL.MSE,
            LOG_GROWTH_MODEL.RMSE
        ]

        for col in expected_columns:
            assert col in results.columns

        # Check that results are reasonable (not all NaN)
        assert not results[LOG_GROWTH_MODEL.R_FIT].isna().all()
        assert not results[LOG_GROWTH_MODEL.K_FIT].isna().all()
        assert not results[LOG_GROWTH_MODEL.N0_FIT].isna().all()

        # Check that growth rate is calculated correctly (r * K / 4)
        r_values = results[LOG_GROWTH_MODEL.R_FIT]
        k_values = results[LOG_GROWTH_MODEL.K_FIT]
        growth_rate_calc = (r_values*k_values)/4
        pd.testing.assert_series_equal(
                results[LOG_GROWTH_MODEL.GROWTH_RATE],
                growth_rate_calc,
                check_names=False
        )

    def test_analyze_with_kmax_label(self, sample_data):
        """Test model analysis with Kmax label specified."""
        # Add a Kmax column to the sample data
        sample_data_with_kmax = sample_data.copy()
        sample_data_with_kmax['Kmax_Value'] = 1200  # Higher than actual max

        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain'],
                Kmax_label='Kmax_Value'
        )

        results = model.analyze(sample_data_with_kmax)

        # Check that K_max values match the specified column
        assert (results[LOG_GROWTH_MODEL.K_MAX] == 1200).all()

    def test_parallel_processing(self, sample_data):
        """Test parallel processing functionality."""
        model_parallel = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain'],
                n_jobs=2
        )

        results_parallel = model_parallel.analyze(sample_data)

        # Compare with single-threaded results
        model_single = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain'],
                n_jobs=1
        )

        results_single = model_single.analyze(sample_data)

        # Results should be very similar (allowing for numerical differences)
        pd.testing.assert_frame_equal(
                results_parallel.sort_index(),
                results_single.sort_index(),
                atol=1e-10
        )

    def test_results_method(self, sample_data):
        """Test the results() method."""
        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain']
        )

        # Should return empty DataFrame if analyze hasn't been called
        empty_results = model.results()
        assert isinstance(empty_results, pd.DataFrame)
        assert empty_results.empty

        # After analyze, should return results
        model.analyze(sample_data)
        results = model.results()

        assert isinstance(results, pd.DataFrame)
        assert not results.empty

    def test_show_method(self, sample_data):
        """Test the show() method for visualization."""
        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain']
        )

        model.analyze(sample_data)

        # Test basic plotting with criteria
        fig, ax = model.show(criteria={'Metadata_Dataset': 'Test'})
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Test plotting with more specific criteria
        fig2, ax2 = model.show(criteria={'Metadata_Dataset': 'Test', 'Metadata_Strain': 'Strain1'})
        assert isinstance(fig2, plt.Figure)
        assert isinstance(ax2, plt.Axes)

        plt.close('all')  # Clean up figures

    def test_show_with_filtered_data(self, sample_data):
        """Test show method with filtered criteria."""
        # Create data with multiple strains
        multi_strain_data = sample_data.copy()
        strain2_data = sample_data.copy()
        strain2_data['Metadata_Strain'] = 'Strain2'

        combined_data = pd.concat([multi_strain_data, strain2_data], ignore_index=True)

        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain']
        )

        model.analyze(combined_data)

        # Test filtering to show only one strain
        fig, ax = model.show(criteria={'Metadata_Strain': 'Strain1'})
        assert isinstance(fig, plt.Figure)

        plt.close('all')

    def test_model_function(self):
        """Test the static model function."""
        # Test basic functionality
        t = np.array([0, 1, 2, 5])
        r, K, N0 = 0.5, 1000, 50

        result = LogGrowthModel._model_func(t, r, K, N0)

        # Check initial condition
        assert abs(result[0] - N0) < 1e-10

        # Check that growth approaches K
        assert result[-1] < K  # Should approach but not reach K
        assert result[-1] > result[0]  # Should increase

        # Test with scalar input
        scalar_result = LogGrowthModel._model_func(1.0, r, K, N0)
        assert isinstance(scalar_result, (float, np.floating))

    def test_loss_function(self):
        """Test the loss function."""
        r, K, N0 = 0.5, 1000, 50
        params = [r, K, N0]

        t = np.array([0, 1, 2])
        y = LogGrowthModel._model_func(t, r, K, N0)
        lam, alpha = 1.0, 1.0

        loss = LogGrowthModel._loss_func(params, t, y, lam, alpha)

        # Loss should be array-like
        assert isinstance(loss, np.ndarray)

        # For perfect fit (y_pred = y), residuals should be zero
        # But regularization and penalty terms will still contribute
        assert len(loss) > len(y)  # Should include regularization terms

    @pytest.mark.parametrize("missing_column", ['Shape_Area', 'Metadata_Time'])
    def test_missing_columns(self, sample_data, missing_column):
        """Test behavior with missing required columns."""
        data_missing_col = sample_data.drop(columns=[missing_column])

        model = LogGrowthModel(
                on='Shape_Area' if missing_column != 'Shape_Area' else 'Missing_Column',
                groupby=['Metadata_Dataset', 'Metadata_Strain']
        )

        with pytest.raises(KeyError):
            model.analyze(data_missing_col)

    def test_empty_data(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['Metadata_Time', 'Shape_Area', 'Metadata_Dataset', 'Metadata_Strain'])

        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain']
        )

        with pytest.raises((ValueError, KeyError)):
            model.analyze(empty_df)

    def test_single_timepoint(self):
        """Test behavior with insufficient data (single timepoint)."""
        single_point_df = pd.DataFrame({
            'Metadata_Time'   : [0],
            'Shape_Area'      : [100],
            'Metadata_Dataset': ['Test'],
            'Metadata_Strain' : ['Strain1']
        })

        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain']
        )

        # Should handle gracefully, possibly returning NaN values
        results = model.analyze(single_point_df)

        # Results should exist but may contain NaN due to insufficient data
        assert isinstance(results, pd.DataFrame)

    def test_real_data_integration(self, measurement_data):
        """Test with real measurement data (subset)."""
        # Use a subset of real data for testing
        real_subset = measurement_data[
            (measurement_data['Metadata_Condition'] == '30C') &
            (measurement_data['Metadata_Strain'] == 'CBS11445')
            ].copy()

        if not real_subset.empty:
            model = LogGrowthModel(
                    on='Shape_Area',
                    groupby=['Metadata_Condition', 'Metadata_Strain'],
                    verbose=False
            )

            results = model.analyze(real_subset)

            assert isinstance(results, pd.DataFrame)
            assert not results.empty

            # Check that fitted parameters are reasonable
            assert results[LOG_GROWTH_MODEL.R_FIT].notna().any()
            assert results[LOG_GROWTH_MODEL.K_FIT].notna().any()
            assert results[LOG_GROWTH_MODEL.N0_FIT].notna().any()

    def test_verbose_output(self, sample_data, capsys):
        """Test verbose output functionality."""
        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain'],
                verbose=True
        )

        model.analyze(sample_data)

        # Check that verbose output was produced
        captured = capsys.readouterr()
        # Note: scipy's verbose output may not always appear in captured stdout
        # This test mainly ensures verbose=True doesn't break anything

    def test_fitting_bounds(self):
        """Test that fitting respects bounds."""
        # Create data with reasonable initial conditions
        t = np.linspace(0, 10, 20)
        # Reasonable initial size
        N0_reasonable = 10
        y = 1000/(1 + (1000 - N0_reasonable)/N0_reasonable*np.exp(-0.5*t))

        df = pd.DataFrame({
            'Metadata_Time'   : t,
            'Shape_Area'      : y,
            'Metadata_Dataset': ['Test']*len(t),
            'Metadata_Strain' : ['Strain1']*len(t)
        })

        model = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain']
        )

        results = model.analyze(df)

        # Check that fitting succeeded (not NaN)
        assert not np.isnan(results[LOG_GROWTH_MODEL.R_FIT].iloc[0])
        assert not np.isnan(results[LOG_GROWTH_MODEL.K_FIT].iloc[0])
        assert not np.isnan(results[LOG_GROWTH_MODEL.N0_FIT].iloc[0])

        # Check that fitted N0 is not below minimum bound (0)
        assert results[LOG_GROWTH_MODEL.N0_FIT].iloc[0] >= 0

        # Check that fitted r is within bounds
        assert results[LOG_GROWTH_MODEL.R_FIT].iloc[0] >= 1e-5

    def test_aggregation_functionality(self, sample_data):
        """Test different aggregation functions."""
        # Test with mean aggregation (default)
        model_mean = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain'],
                agg_func='mean'
        )

        results_mean = model_mean.analyze(sample_data)

        # Test with median aggregation
        model_median = LogGrowthModel(
                on='Shape_Area',
                groupby=['Metadata_Dataset', 'Metadata_Strain'],
                agg_func='median'
        )

        results_median = model_median.analyze(sample_data)

        # Both should produce results
        assert isinstance(results_mean, pd.DataFrame)
        assert isinstance(results_median, pd.DataFrame)

        # Results may differ due to different aggregation
        assert not results_mean.empty
        assert not results_median.empty

"""Unit tests for dataset upload functionality."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import click

from psyctl.core.dataset_builder import DatasetBuilder


class TestValidateHfToken:
    """Test HF_TOKEN validation."""

    def test_validate_hf_token_success(self, monkeypatch):
        """Test HF_TOKEN validation with valid token."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_1234567890")
        from psyctl.core.utils import validate_hf_token

        token = validate_hf_token()
        assert token == "hf_test_token_1234567890"

    def test_validate_hf_token_missing(self, monkeypatch):
        """Test HF_TOKEN validation with missing token."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        from psyctl.core.utils import validate_hf_token

        with pytest.raises(click.ClickException) as exc_info:
            validate_hf_token()

        assert "HF_TOKEN environment variable is required" in str(exc_info.value)
        assert "huggingface.co/settings/tokens" in str(exc_info.value)


class TestDatasetCard:
    """Test dataset card generation."""

    def test_generate_dataset_card(self):
        """Test dataset card generation with PSYCTL branding."""
        builder = DatasetBuilder()

        card = builder._generate_dataset_card(
            personality="Extroversion",
            model="google/gemma-3-27b-it",
            num_samples=1000,
            timestamp="2025-01-07T14:30:00"
        )

        # Check PSYCTL branding
        assert "PSYCTL" in card
        assert "https://cdn.caveduck.io" in card  # Logo URL
        assert "Generated with [PSYCTL]" in card
        assert "Extroversion" in card
        assert "1000" in card
        assert "google/gemma-3-27b-it" in card

    def test_dataset_card_personality_tag(self):
        """Test dataset card includes personality tag."""
        builder = DatasetBuilder()

        card = builder._generate_dataset_card(
            personality="Machiavellianism",
            model="test-model",
            num_samples=500,
            timestamp="2025-01-07T14:30:00"
        )

        assert "machiavellianism" in card.lower()
        assert "- machiavellianism" in card


class TestUploadToHub:
    """Test upload to HuggingFace Hub."""

    @patch('psyctl.core.dataset_builder.Dataset')
    @patch('psyctl.core.dataset_builder.HfApi')
    @patch('builtins.open', new_callable=mock_open, read_data='{"question": "test", "positive": "(1", "neutral": "(2"}\n')
    def test_upload_to_hub_success(self, mock_file, mock_hf_api, mock_dataset, tmp_path):
        """Test successful upload to HuggingFace Hub."""
        # Setup mocks
        mock_ds_instance = MagicMock()
        mock_ds_instance.__len__ = Mock(return_value=1)
        mock_dataset.from_list.return_value = mock_ds_instance

        builder = DatasetBuilder()
        builder.personality = "Extroversion"
        builder.active_model = "google/gemma-3-27b-it"

        # Create test JSONL file
        test_file = tmp_path / "test_dataset.jsonl"
        test_file.write_text('{"question": "test", "positive": "(1", "neutral": "(2"}\n')

        repo_url = builder.upload_to_hub(
            jsonl_file=test_file,
            repo_id="test-user/test-repo",
            private=False,
            token="hf_test_token"
        )

        assert repo_url == "https://huggingface.co/datasets/test-user/test-repo"
        mock_ds_instance.push_to_hub.assert_called_once()

    def test_upload_invalid_repo_id(self, tmp_path):
        """Test upload with invalid repo_id format."""
        builder = DatasetBuilder()

        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"question": "test"}\n')

        with pytest.raises(ValueError) as exc_info:
            builder.upload_to_hub(
                jsonl_file=test_file,
                repo_id="invalid-repo-id",  # Missing username/
                token="hf_test"
            )

        assert "Invalid repo_id format" in str(exc_info.value)
        assert "username/repo-name" in str(exc_info.value)

    def test_upload_file_not_found(self):
        """Test upload with non-existent file."""
        builder = DatasetBuilder()

        with pytest.raises(FileNotFoundError):
            builder.upload_to_hub(
                jsonl_file=Path("/non/existent/file.jsonl"),
                repo_id="test-user/test-repo",
                token="hf_test"
            )

    @patch('psyctl.core.dataset_builder.Dataset')
    @patch('psyctl.core.dataset_builder.HfApi')
    def test_upload_creates_readme(self, mock_hf_api, mock_dataset, tmp_path):
        """Test that upload creates README.md with dataset card."""
        # Setup mocks
        mock_ds_instance = MagicMock()
        mock_ds_instance.__len__ = Mock(return_value=100)
        mock_dataset.from_list.return_value = mock_ds_instance

        builder = DatasetBuilder()
        builder.personality = "Extroversion"
        builder.active_model = "google/gemma-3-27b-it"

        # Create test JSONL file
        test_file = tmp_path / "test_dataset.jsonl"
        test_file.write_text('{"question": "test", "positive": "(1", "neutral": "(2"}\n')

        builder.upload_to_hub(
            jsonl_file=test_file,
            repo_id="test-user/test-repo",
            private=False,
            token="hf_test_token"
        )

        # Check README was created
        readme_path = tmp_path / "README.md"
        assert readme_path.exists()

        # Check README content
        readme_content = readme_path.read_text()
        assert "PSYCTL" in readme_content
        assert "Extroversion" in readme_content


class TestUploadCLI:
    """Test dataset.upload CLI command."""

    @patch('psyctl.commands.dataset.DatasetBuilder')
    @patch('psyctl.commands.dataset.validate_hf_token')
    def test_upload_cli_success(self, mock_validate_token, mock_builder_class, tmp_path):
        """Test successful CLI upload."""
        from click.testing import CliRunner
        from psyctl.commands.dataset import upload

        # Setup mocks
        mock_validate_token.return_value = "hf_test_token"
        mock_builder = MagicMock()
        mock_builder.upload_to_hub.return_value = "https://huggingface.co/datasets/test-user/test-repo"
        mock_builder_class.return_value = mock_builder

        # Create test file
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"question": "test"}\n')

        runner = CliRunner()
        result = runner.invoke(upload, [
            '--dataset-file', str(test_file),
            '--repo-id', 'test-user/test-repo'
        ])

        assert result.exit_code == 0
        assert "Successfully uploaded" in result.output
        mock_builder.upload_to_hub.assert_called_once()

    @patch('psyctl.commands.dataset.validate_hf_token')
    def test_upload_cli_missing_token(self, mock_validate_token):
        """Test CLI upload with missing HF_TOKEN."""
        from click.testing import CliRunner
        from psyctl.commands.dataset import upload

        # Mock token validation to raise exception
        mock_validate_token.side_effect = click.ClickException("HF_TOKEN not found")

        runner = CliRunner()
        result = runner.invoke(upload, [
            '--dataset-file', 'test.jsonl',
            '--repo-id', 'test-user/test-repo'
        ])

        assert result.exit_code != 0
        assert "HF_TOKEN" in result.output

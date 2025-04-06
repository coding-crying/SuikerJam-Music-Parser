# Jam Session Analyzer

A Python tool to automatically analyze long jam session recordings and detect:
- Tempo changes
- Song boundaries (through structural analysis)
- Silence sections (potential song boundaries)
- Clapping/applause sections
- Speech/conversation sections
- High-confidence song boundaries (combining multiple indicators)

This tool is designed to handle very long audio files (multiple hours) by processing them in manageable chunks.

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python jam_session_analyzer.py path/to/your/jam_session.wav
```

This will analyze the audio file and print the detected events to the console.

### Saving results to a file

To save the results to a JSON file:

```bash
python jam_session_analyzer.py path/to/your/jam_session.wav --output results.json
```

### Advanced options

You can customize the analysis parameters:

```bash
python jam_session_analyzer.py path/to/your/jam_session.wav \
  --chunk-duration 300 \  # Duration of each analysis chunk in seconds (default: 300)
  --overlap 5 \           # Overlap between chunks in seconds (default: 5)
  --tempo-threshold 5.0 \ # Minimum BPM difference for tempo change detection (default: 5)
  --boundary-threshold 0.15 \ # Threshold for structural boundary detection (default: 0.15)
  --silence-threshold -50 \   # Silence threshold in dB (default: -50)
  --silence-duration 2.0 \    # Minimum silence duration in seconds (default: 2)
  --min-segment 30 \          # Minimum segment length in seconds (default: 30)
  --clapping-duration 1.0 \   # Minimum clapping duration in seconds (default: 1.0)
  --clapping-percentile 20 \  # Energy percentile for clapping detection (default: 20)
  --clapping-irregularity 0.6 \ # Irregularity threshold for clapping (default: 0.6)
  --speech-duration 1.0 \     # Minimum speech duration in seconds (default: 1.0)
  --proximity-threshold 30 \  # Max time between related events for song boundary detection (default: 30)
  --no-song-boundaries \      # Disable high-confidence song boundary identification
  --quiet                     # Reduce output verbosity
```

## How It Works

The analyzer uses five complementary approaches to analyze jam sessions:

1. **Tempo Analysis**: Detects significant changes in tempo (BPM) which often indicate new sections or songs.
2. **Structural Segmentation**: Uses music features (timbre and harmony) to find points where the musical content changes significantly.
3. **Silence Detection**: Identifies periods of silence which often indicate breaks between songs.
4. **Clapping Detection**: Identifies sections with audience applause/clapping based on audio characteristics.
5. **Speech Detection**: Identifies sections with human conversation, which often happens between songs.

Additionally, the analyzer identifies **high-confidence song boundaries** by looking for patterns of multiple indicators occurring close together in time.

### Analysis Process

1. The audio file is processed in overlapping chunks to handle very long recordings efficiently.
2. For each chunk:
   - Tempo is estimated using beat tracking
   - Musical features (MFCCs and chroma) are extracted and analyzed for structural changes
   - RMS energy is analyzed to detect silence sections
   - Multiple audio features are analyzed to detect clapping sections
   - Speech characteristics are analyzed to detect conversation sections
3. All events are combined and analyzed to identify high-confidence song boundaries.
4. Results are sorted and filtered to produce the final event timestamps.

### Clapping Detection

The clapping detector uses multiple audio characteristics to identify applause sections:

- **Spectral Features**: Clapping has more high-frequency content (higher spectral centroid) and noise-like qualities (higher spectral flatness) compared to most musical content
- **Rhythmic Irregularity**: Clapping typically has less regular periodicity than musical beats
- **Energy Range**: Clapping is usually quieter than the main musical content but louder than silence
- **Transient Detection**: Clapping has distinctive sharp, irregular transients
- **Frequency Distribution**: Clapping has higher mid-high to low frequency energy ratio compared to drums

Each detected clapping section includes a confidence score (0-1) indicating how likely it is to be actual applause.

### Speech Detection

The speech detector identifies sections likely to contain human conversation:

- **Spectral Characteristics**: Human speech has distinctive spectral shapes with energy concentrated in certain frequency bands
- **Formant Structure**: The detector analyzes the relationship between frequency bands typical of vocal formants
- **MFCC Dynamics**: Speech has characteristic patterns in MFCC coefficients and their variations over time
- **Syllabic Rhythm**: Speech tends to have energy fluctuations in the 4-8 Hz range, corresponding to syllable rate
- **Energy Range**: Speech is usually at moderate energy levels in jam sessions

### Song Boundary Identification

The song boundary detector looks for patterns that strongly indicate transitions between songs:

- **Silence + Speech Pattern**: When silence is followed by conversation, often indicating a song ending
- **Clapping + Speech/Silence**: Applause after a song, often followed by conversation or silence
- **Multiple Indicators**: The highest confidence is given when multiple indicators (silence, speech, clapping) occur close together
- **Tempo Changes**: When a significant tempo change occurs after silence or speech, likely indicating a new song starting

Each identified song boundary includes a confidence score and description of the evidence that led to its detection.

## Tuning Parameters

For best results, you may need to tune the parameters based on your specific recordings:

- `--tempo-threshold`: Lower values detect more subtle tempo changes
- `--boundary-threshold`: Lower values detect more potential song boundaries (may include false positives)
- `--silence-threshold`: Adjust based on the noise floor of your recording
- `--min-segment`: Prevents detecting boundaries too close together
- `--clapping-percentile`: Lower values are more sensitive to quieter clapping
- `--clapping-irregularity`: Higher values make the clapping detector more sensitive
- `--clapping-duration`: Minimum duration for a section to be considered clapping
- `--speech-duration`: Minimum duration for a section to be considered speech
- `--proximity-threshold`: Maximum time between events to be considered related for song boundary detection

## Output Format

The JSON output contains:
- File metadata
- Analysis parameters used
- List of detected events with timestamps and types

### Event Types

- **Tempo Change**: Includes BPM before and after the change
- **Structural Boundary**: Marks substantial changes in musical content
- **Silence Boundary**: Marks significant silent sections with duration
- **Clapping**: Marks sections with detected applause, including start and end times, duration, and confidence score
- **Speech**: Marks sections with detected conversation, including start and end times, duration, and confidence score
- **Song Boundary**: High-confidence song boundaries with confidence scores and supporting evidence

## Requirements

- Python 3.6+
- librosa (audio analysis)
- numpy (numerical processing)
- soundfile (audio file handling)

## License

MIT 
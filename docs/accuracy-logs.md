# accuracy metrics: WER on vietnamese test set

targeting: word error rate (WER) &lt;5%

manually evaluate WER using `transformers` + `jiwer`, value may differ from official value in original paper

always specify language, whisper lang-auto-detect capability sucks!

once LoRA trained, should merge into base model and run inference @ `float16` because `int8` much slower

`wav2vec bert v2` is faster than `whisper-tiny` on inference but somehow require much more VRAM when fine-tuning, i can only fit batch size = 1 on free tier gpu

try speculative decoding / assistant model but fail with error

use default greedy search, because beam search trigger a spike in VRAM which may cause out-of-memory on free tier gpu

<table>
	<thead>
		<tr>
			<th>evaluate WER on test set<br />model @ <code>float16</code></th>
			<th>CommonVoice&nbsp;v16.1<br />1326 samples</th>
			<th>FLEURS<br />857&nbsp;samples</th>
			<th>VIVOS<br />760&nbsp;samples</th>
			<th>Bud500<br />7500&nbsp;samples</th>
			<th>LSVSC<br />5683&nbsp;samples</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>original whisper large v3<br /><a href="https://huggingface.co/openai/whisper-large-v3">https://huggingface.co/openai/whisper-large-v3</a></td>
			<td>16.2%</td>
			<td>8.3%</td>
			<td>12.3%</td>
			<td></td>
			<td></td>
		</tr>
		<tr>
			<td>🡼 (my) LoRA trained on uncleaned combined datasets<br /><a href="https://huggingface.co/doof-ferb/whisper-large-peft-lora-vi">https://huggingface.co/doof-ferb/whisper-large-peft-lora-vi</a><br /><em>N.B.</em> my LoRA is 3× bigger than others</td>
			<td>14.7%</td>
			<td>14.7%</td>
			<td>9.4%</td>
			<td></td>
			<td></td>
		</tr>
		<tr>
			<td>original whisper large v2<br /><a href="https://huggingface.co/openai/whisper-large-v2">https://huggingface.co/openai/whisper-large-v2</a></td>
			<td>19%</td>
			<td>11.9%</td>
			<td>14.7%</td>
			<td></td>
			<td></td>
		</tr>
		<tr>
			<td>🡼 large v2 traditionally fine-tuned by VinAI<br /><a href="https://huggingface.co/vinai/PhoWhisper-large">https://huggingface.co/vinai/PhoWhisper-large</a></td>
			<td>8.1%</td>
			<td>8.7%</td>
			<td>4.6%</td>
			<td>9.2%</td>
			<td>9.9%</td>
		</tr>
		<tr>
			<td>original whisper medium<br /><a href="https://huggingface.co/openai/whisper-medium">https://huggingface.co/openai/whisper-medium</a></td>
			<td>23.3%</td>
			<td>13.4%</td>
			<td>16.3%</td>
			<td><em>&gt;100%</em> ??</td>
			<td>15.3%</td>
		</tr>
		<tr>
			<td>original whisper small<br /><a href="https://huggingface.co/openai/whisper-small">https://huggingface.co/openai/whisper-small</a></td>
			<td>30.2%</td>
			<td>22.4%</td>
			<td>26.6%</td>
			<td>88.7%</td>
			<td>19.8%</td>
		</tr>
		<tr>
			<td>original whisper tiny<br /><a href="https://huggingface.co/openai/whisper-tiny">https://huggingface.co/openai/whisper-tiny</a></td>
			<td><em>&gt;100%</em> ??</td>
			<td>88.6%</td>
			<td>62.5%</td>
			<td><em>&gt;100%</em> ??</td>
			<td>67.4%</td>
		</tr>
		<tr>
			<td>🡼 (my) whisper tiny fine-tuned on uncleaned combined datasets<br /><a href="https://huggingface.co/doof-ferb/whisper-tiny-vi">https://huggingface.co/doof-ferb/whisper-tiny-vi</a></td>
			<td>26.6%</td>
			<td>37.1%</td>
			<td>18.7%</td>
			<td>9.2%</td>
			<td>17.4%</td>
		</tr>
		<tr>
			<td>original wav2vec bert v2</strong><br /><a href="https://huggingface.co/facebook/w2v-bert-2.0">https://huggingface.co/facebook/w2v-bert-2.0</a></td>
			<td colspan="5"><em>no ready-to-use code, the demo example throw error</em></td>
		</tr>
		<tr>
			<td>🡼 someone fine-tuned on CommonVoice v16.0<br /><a href="https://huggingface.co/trick4kid/w2v-bert-2.0-vietnamese-CV16.0">https://huggingface.co/trick4kid/w2v-bert-2.0-vietnamese-CV16.0</a></td>
			<td>32.8%</td>
			<td>53.8%</td>
			<td>39.3%</td>
			<td></td>
			<td></td>
		</tr>
	</tbody>
</table>

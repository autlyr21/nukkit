<script lang="ts">
	import init, { u2net_preprocess, u2net_postprocess } from '$lib/pkg/front_wasm';
	import * as ort from 'onnxruntime-web';
	import { sleep, openDB, fetchModel, preventImageRedirect } from '$lib/index';
	import { onMount } from 'svelte';

	interface PreProcessed {
		orgPng: Uint8Array;
		inputBuffer: Float32Array;
		nW: number;
		nH: number;
	}

	//Binded Elements
	let imgInput: HTMLInputElement;
	let dropLabel: HTMLLabelElement;
	let inferenceButton: HTMLButtonElement;

	//Page State
	let imgData = $state<string | undefined>(undefined);
	let nukkyData = $state<string | undefined>(undefined);
	let isIdle = $state<boolean>(true);
	let isLoading = $state<boolean>(true);
	const isDisabled = $derived<boolean>(isLoading || !isIdle || !imgData);

	const handleImgInput = (e: Event) => {
		if (!isIdle) return;
		if (e.target.files.length === 0) return;
		const fileType = e.target.files[0].type;
		if (
			fileType.split('/')[0] != 'image' ||
			!['jpeg', 'jpg', 'png', 'webp'].includes(fileType.split('/')[1])
		) {
			alert('유효한 파일을 드롭해 주세요.(.jpg, .jpeg, .png, .webp)');
			return;
		}
		const file = URL.createObjectURL(e.target.files[0]);
		if (imgData) {
			const prevData = imgData;
			imgData = undefined;
			URL.revokeObjectURL(prevData.toString());
		}
		imgData = file;
		nukkyData = undefined;
	};
	const handleImgDragNDrop = async (e: DragEvent) => {
		e.preventDefault();
		if (!isIdle) return;
		const dt = e.dataTransfer;
		if (!dt) return;
		const files = dt.files;
		if (!files) return;
		const fileType = e.dataTransfer.files[0].type;
		if (
			fileType.split('/')[0] != 'image' ||
			!['jpeg', 'jpg', 'png', 'webp'].includes(fileType.split('/')[1])
		) {
			alert('유효한 파일을 드롭해 주세요.(.jpg, .jpeg, .png, .webp)');
			return;
		}
		const file = e.dataTransfer.files[0];
		const url = URL.createObjectURL(file.slice());
		if (imgData) {
			const prevData = imgData;
			imgData = undefined;
			URL.revokeObjectURL(prevData);
		}
		imgData = url;
		nukkyData = undefined;
	};
	const initInferenceSession = async (db: IDBDatabase) => {
		// 아이폰은 쓰레딩 활성화 시 out of memory 에러 발생 20240310
		if (navigator.userAgent.toLowerCase().includes('iphone')) {
			ort.env.wasm.numThreads = 1;
		}
		const onnxBuffer = await fetchModel(db, 'models', 'u2netp');
		const session = await ort.InferenceSession.create(onnxBuffer);
		return session;
	};

	onMount(async () => {
		preventImageRedirect();
		const db: IDBDatabase = await openDB('nukkit', 1, 'models');
		imgInput.addEventListener('change', handleImgInput);
		dropLabel.addEventListener('drop', async (e) => {
			handleImgDragNDrop(e);
		});
		const [session, _] = await Promise.all([initInferenceSession(db), init()]);

		const preprocess = async (): Promise<PreProcessed> => {
			const tempImg = new Image();
			tempImg.src = imgData;
			await tempImg.decode();
			const nW = tempImg.naturalWidth;
			const nH = tempImg.naturalHeight;

			const tempCanvas = document.createElement('canvas');
			const tempCtx = tempCanvas.getContext('2d');
			tempCanvas.width = nW;
			tempCanvas.height = nH;
			tempCtx?.drawImage(tempImg, 0, 0);
			const pngBlob: Blob = await new Promise((resolve) => {
				tempCanvas.toBlob(resolve, 'image/png');
			});
			const orgPng = new Uint8Array(await pngBlob.arrayBuffer());
			const f32Buffer = u2net_preprocess(orgPng);

			return { orgPng, inputBuffer: f32Buffer };
		};

		const infer = async (buffer: Float32Array): Promise<Float32Array> => {
			const input = new ort.Tensor('float32', Float32Array.from(buffer), [1, 3, 320, 320]);
			const feeds = { 'input.1': input };
			const res = await session.run(feeds);
			const mask = res['1959'].cpuData;
			return mask;
		};

		const postproess = async (mask: Float32Array, orgPng: Uint8Array, nW: number, nH: number) => {
			const nukky = u2net_postprocess(orgPng, mask);

			const clamped = Uint8ClampedArray.from(nukky);
			const newImgData = new ImageData(clamped, nW, nH);
			const tempCanvas = document.createElement('canvas');
			const tempCtx = tempCanvas.getContext('2d');
			tempCanvas.width = nW;
			tempCanvas.height = nH;
			tempCtx?.putImageData(newImgData, 0, 0);
			nukkyData = tempCanvas.toDataURL();
		};
		const handleDownload = (dataUrl: string) => {
			const tempA = document.createElement('a');
			tempA.href = dataUrl;
			tempA.download = 'nukky.png';
			document.body.appendChild(tempA);
			tempA.click();
			document.body.removeChild(tempA);
		};
		const handleInference = async () => {
			const tempImg = new Image();
			tempImg.src = imgData;
			await tempImg.decode();

			const nW = tempImg.naturalWidth;
			const nH = tempImg.naturalHeight;

			if (nW * nH > 16777216) {
				alert('이미지가 너무 큽니다. 해상도가 더 낮은 이미지를 선택해 주세요.');
				return;
			}
			isIdle = false;
			await sleep(500);
			const { orgPng, inputBuffer } = await preprocess();
			const mask = await infer(inputBuffer);
			await postproess(mask, orgPng, nW, nH);
			isIdle = true;
		};

		const handleMainButton = async () => {
			if (!isIdle) return;
			if (nukkyData) handleDownload(nukkyData);
			else await handleInference();
		};
		inferenceButton.addEventListener('click', handleMainButton);
		isLoading = false;
	});
</script>

<div class="grid w-[100dvw] h-[100dvh] p-4 gap-4 place-items-center">
	<div class="flex flex-col place-items-center gap-4">
		<h1 class="font-extrabold text-5xl text-gray-100">Nukkit!</h1>
		<div class="flex flex-col place-items-center text-gray-300">
			<h2 class="font-semibold text-xl">AI-Powered, Fast, Local</h2>
			<h2 class="font-semibold text-xl">Background Remover</h2>
		</div>
		<h2 class="font-semibold text-sm text-center text-gray-300">
			Remove bg without sending your images to server.
		</h2>
		<label
			bind:this={dropLabel}
			for="dropzone-file"
			class="flex flex-col items-center justify-center w-80 min-h-20 lg:w-96 lg:min-h-20 overflow-hidden border-2 border-dashed rounded-lg cursor-pointer"
		>
			{#if imgData}
				<div class="grid relative w-full h-full checkerboard">
					<img
						src={imgData}
						class={nukkyData
							? 'object-cover w-80 lg:w-96 duration-1000 opacity-0'
							: 'object-cover w-80 lg:w-96 duration-1000 opacity-100'}
						alt="selected"
					/>
					{#if nukkyData}
						<img src={nukkyData} class="absolute top-0 object-cover w-80 lg:w-96" alt="nukky" />
					{/if}
				</div>
			{:else}
				<div class="flex flex-col w-80 h-80 lg:w-96 lg:h-96 items-center justify-center pt-5 pb-6">
					<svg
						class="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
						xmlns="http://www.w3.org/2000/svg"
						width="40"
						height="40"
						viewBox="0 0 24 24"
						><path
							stroke="currentColor"
							fill="currentColor"
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M14 9l-2.519 4-2.481-1.96-5 6.96h16l-6-9zm8-5v16h-20v-16h20zm2-2h-24v20h24v-20zm-20 6c0-1.104.896-2 2-2s2 .896 2 2c0 1.105-.896 2-2 2s-2-.895-2-2z"
						/></svg
					>
					<p class="mb-2 text-sm text-gray-500 dark:text-gray-400">
						<span class="font-semibold">Click to select</span> or drag and drop
					</p>
					<p class="text-xs text-gray-500 dark:text-gray-400">PNG, JPG or WEBP</p>
				</div>
			{/if}
			<input
				accept="image/jpeg, image/png, image/webp"
				bind:this={imgInput}
				id="dropzone-file"
				type="file"
				class="hidden"
			/>
		</label>
		<button
			bind:this={inferenceButton}
			class={isDisabled
				? 'btn btn-accent w-80 lg:w-96 btn-disabled text-gray-100'
				: 'btn btn-accent w-80 lg:w-96'}
		>
			{#if isLoading}
				Loading Model...
			{:else if isIdle}
				{nukkyData ? 'Save' : 'Nukkit!'}
			{:else}
				Processing...
			{/if}
		</button>
	</div>
</div>

<style>
	.checkerboard {
		background: conic-gradient(
			silver 90deg,
			white 90deg 180deg,
			silver 180deg 270deg,
			white 270deg 360deg
		);
		background-repeat: repeat;
		background-size: 20px 20px;
		background-position: top left;
	}
</style>

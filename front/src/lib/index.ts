// place files you want to import through the `$lib` alias in this folder.
export const sleep = async (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
export const openDB = async (
	dbName: string,
	version: number,
	storeName: string
): Promise<IDBDatabase> => {
	return new Promise((resolve, reject) => {
		const req = indexedDB.open(dbName, version);
		req.onupgradeneeded = (event: any) => {
			const db = event.target.result;
			if (!db.objectStoreNames.contains(storeName)) db.createObjectStore(storeName);
		};
		req.onerror = (event: any) => {
			reject(event.target.error);
		};
		req.onsuccess = (event: any) => {
			resolve(event.target.result);
		};
	});
};
const getModel = async (
	db: IDBDatabase,
	storeName: string,
	modelName: string
): Promise<undefined | ArrayBuffer> => {
	return new Promise((resolve, reject) => {
		const transaction = db.transaction([storeName], 'readonly');
		const store = transaction.objectStore(storeName);
		const req = store.get(modelName);
		req.onsuccess = (event: any) => resolve(event.target.result);
		req.onerror = (event: any) => reject(event.target.error);
	});
};
const putModel = async (
	db: IDBDatabase,
	storeName: string,
	modelName: string,
	buffer: ArrayBuffer
) => {
	return new Promise((resolve, reject) => {
		const transaction = db.transaction([storeName], 'readwrite');
		const store = transaction.objectStore(storeName);
		const req = store.put(buffer, modelName);
		req.onsuccess = (event: any) => resolve('');
		req.onerror = (event: any) => reject(event.target.error);
	});
};

export const fetchModel = async (
	db: IDBDatabase,
	storeName: string,
	modelName: string
): Promise<ArrayBuffer> => {
	const storedModel = await getModel(db, storeName, modelName);
	if (storedModel) return storedModel;

	const onnxFile = await fetch(`/models/${modelName}.onnx`, { cache: 'force-cache' });
	const onnxBuffer = await onnxFile.arrayBuffer();
	await putModel(db, storeName, modelName, onnxBuffer);
	return onnxBuffer;
};

export const preventImageRedirect = () => {
	window.addEventListener('dragover', (e) => {
		e.preventDefault();
	});
	window.addEventListener('drop', (e) => {
		e.preventDefault();
	});
};

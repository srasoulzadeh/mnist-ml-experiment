using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;

public class MnistSample : MonoBehaviour
{
    public NNModel modelAsset;
    private Model m_RuntimeModel;
    private IWorker m_Worker;

    [SerializeField] Text outputTextView = null;
    [SerializeField] ComputeShader compute = null;

    bool isProcessing = false;
    float[,] inputs = new float[28, 28];
    float[] outputs = new float[10];
    ComputeBuffer inputBuffer;

    
    System.Text.StringBuilder sb = new System.Text.StringBuilder();

    private 

    void Start()
    {
        this.m_RuntimeModel = ModelLoader.Load(modelAsset);
        this.m_Worker = WorkerFactory.CreateWorker(m_RuntimeModel);

        inputBuffer = new ComputeBuffer(28 * 28, sizeof(float));
    }

    void OnDestroy()
    {
        inputBuffer?.Dispose();
    }

    public void OnDrawTexture(RenderTexture texture)
    {
        if (!isProcessing)
        {
            isProcessing = true;

            compute.SetTexture(0, "InputTexture", texture);
            compute.SetBuffer(0, "OutputTensor", inputBuffer);
            compute.Dispatch(0, 28 / 4, 28 / 4, 1);
            inputBuffer.GetData(inputs);

            var input = new Tensor(texture, 1);
            var output = this.m_Worker.Execute(input).PeekOutput().Flatten();

            int predicted_label = output.ArgMax()[0];

            sb.Clear();
            sb.AppendLine($"{predicted_label}");
            outputTextView.text = sb.ToString();

            isProcessing = false;
        }
    }
}

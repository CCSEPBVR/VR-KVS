/*****************************************************************************/
/**
 *  @file   main.cpp
 *  @brief  Example program for texture filtering.
 *  @author Insight, Inc.
 */
#include <kvs/Version>

#include "TexturedPolygonObject.h"
#include "TexturedPolygonImporter.h"
#include "StochasticTexturedPolygonRenderer.h"

//#define USE_OCULUS
#ifdef USE_OCULUS
#include <Application.h>
#include <Screen.h>
#include <TouchController.h>
#else
#include <kvs/glut/Application>
#include <kvs/glut/Screen>
#endif

namespace KVSMain {
    // 点の数
    const static size_t N = 16;
    
    // ポリゴンの数
    const static size_t P = 14;
    
    static kvs::Real32 CoordArray[N * 3] = {
        10,  0, 20, // 点0の座標  (1.0, 0.0, 2.0)
        20,  0, 20, // 点1の座標  (2.0, 0.0, 2.0)
        15,  8, 17, // 点2の座標  (1.5, 0.8, 1.7)
        15,  0, 11, // 点3の座標  (1.5, 0.0, 1.1)
        30,  0, 20, // 点4の座標  (3.0, 0.0, 2.0)
        40,  0, 20, // 点5の座標  (4.0, 0.0, 2.0)
        35,  8, 17, // 点6の座標  (3.5, 0.8, 1.7)
        35,  0, 11, // 点7の座標  (3.5, 0.0, 1.1)
        20,  0, 30, // 点8の座標  (2.0, 0.0, 3.0)
        30,  0, 30, // 点9の座標  (3.0, 0.0, 3.0)
        25,  8, 27, // 点10の座標 (2.5, 0.8, 2.7)
        25,  0, 21, // 点11の座標 (2.5, 0.0, 2.1)
         0, -1,  0, // 点12の座標 (0.0, -0.1, 0.0)
        50, -1,  0, // 点13の座標 (5.0, -0.1, 0.0)
        50, -1, 50, // 点14の座標 (5.0, -0.1, 5.0)
         0, -1, 50  // 点15の座標 (0.0, -0.1, 5.0)
    };
    
    // カラー値配列の定義
    static kvs::UInt8 ColorArray[N * 3] = {
        255,   0,   0, // 点0の色  赤(255,   0,   0)
          0, 255,   0, // 点1の色  緑(  0, 255,   0)
          0,   0, 255, // 点2の色  青(  0,   0, 255)
        255, 255,   0, // 点3の色  黄(255, 255,   0)
        255,   0,   0, // 点4の色  赤(255,   0,   0)
          0, 255,   0, // 点5の色  緑(  0, 255,   0)
          0,   0, 255, // 点6の色  青(  0,   0, 255)
        255, 255,   0, // 点7の色  黄(255, 255,   0)
        255,   0,   0, // 点8の色  赤(255,   0,   0)
          0, 255,   0, // 点9の色  緑(  0, 255,   0)
          0,   0, 255, // 点10の色 青(  0,   0, 255)
        255, 255,   0, // 点11の色 黄(255, 255,   0)
        255,   0,   0, // 点12の色 黄(255,   0,   0)
        255,   0,   0, // 点13の色 黄(255,   0,   0)
        255,   0,   0, // 点14の色 黄(255,   0,   0)
        255,   0,   0  // 点15の色 黄(255,   0,   0)
    };
    
    // 法線ベクトル配列の定義
    static kvs::Real32 NormalArray[N * 3] = {
         0.00f,  0.3f,  0.8f, // 点0に対する法線ベクトル
         0.00f, -0.9f,  0.0f, // 点1に対する法線ベクトル
         0.72f,  0.3f, -0.4f, // 点2に対する法線ベクトル
        -0.72f,  0.3f, -0.4f, // 点3に対する法線ベクトル
         0.00f,  0.3f,  0.8f, // 点4に対する法線ベクトル
         0.00f, -0.9f,  0.0f, // 点5に対する法線ベクトル
         0.72f,  0.3f, -0.4f, // 点6に対する法線ベクトル
        -0.72f,  0.3f, -0.4f, // 点7に対する法線ベクトル
         0.00f,  0.3f,  0.8f, // 点8に対する法線ベクトル
         0.00f, -0.9f,  0.0f, // 点9に対する法線ベクトル
         0.72f,  0.3f, -0.4f, // 点10に対する法線ベクトル
        -0.72f,  0.3f, -0.4f, // 点11に対する法線ベクトル
         0.00f, -1.0f,  0.0f, // 点12に対する法線ベクトル
         0.00f, -1.0f,  0.0f, // 点13に対する法線ベクトル
         0.00f, -1.0f,  0.0f, // 点14に対する法線ベクトル
         0.00f, -1.0f,  0.0f  // 点15に対する法線ベクトル
    };
    
    // ポリゴンの接続情報配列の定義
    static kvs::UInt32 PolygonConnectionArray[P * 3] = {
         0,  1,  2, // 点0, 1, 2を接続
         0,  3,  1, // 点0, 3, 1を接続
         1,  3,  2, // 点1, 3, 2を接続
         0,  2,  3, // 点0, 2, 3を接続
         4,  5,  6, // 点4, 5, 6を接続
         4,  7,  5, // 点4, 7, 5を接続
         5,  7,  6, // 点5, 7, 6を接続
         4,  6,  7, // 点4, 6, 7を接続
         8,  9, 10, // 点8, 9, 10を接続
         8, 11,  9, // 点8, 11, 9を接続
         9, 11, 10, // 点9, 11, 10を接続
         8, 10, 11, // 点8, 10, 11を接続
        12, 13, 14, // 点12, 13, 14を接続
        12, 14, 15  // 点12, 14, 15を接続
    };
    
    // テクスチャ座標の配列の定義
    static kvs::Real32 TextureCoordArray[N * 2] = {
        0.2f, 0.3f,  // 点0に対するテクスチャ座標
        0.6f, 0.3f,  // 点1に対するテクスチャ座標
        0.4f, 0.7f,  // 点2に対するテクスチャ座標
        0.8f, 0.7f,  // 点3に対するテクスチャ座標
        0.8f, 0.7f,  // 点4に対するテクスチャ座標
        0.4f, 0.7f,  // 点5に対するテクスチャ座標
        0.6f, 0.3f,  // 点6に対するテクスチャ座標
        0.2f, 0.3f,  // 点7に対するテクスチャ座標
        0.8f, 0.7f,  // 点8に対するテクスチャ座標
        0.4f, 0.7f,  // 点9に対するテクスチャ座標
        0.6f, 0.3f,  // 点10に対するテクスチャ座標
        0.2f, 0.3f,  // 点11に対するテクスチャ座標
        0.0f, 0.0f,  // 点12に対するテクスチャ座標
        1.0f, 0.0f,  // 点13に対するテクスチャ座標
        1.0f, 1.0f,  // 点14に対するテクスチャ座標
        0.0f, 1.0f   // 点15に対するテクスチャ座標
    };
    
    // テクスチャIDの配列の定義
    static kvs::UInt32 TextureIdArray[N] = {
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3
    };
    
    // テクスチャ画像のサイズ フィルタリングの差異を確認するには、ある程度細かい画像である必要がある
    const static size_t TextureSize = 2048;
    static unsigned char *TextureArray0 = new unsigned char[TextureSize * TextureSize * 4];
    static unsigned char *TextureArray1 = new unsigned char[TextureSize * TextureSize * 4];
    static unsigned char *TextureArray2 = new unsigned char[TextureSize * TextureSize * 4];
    static unsigned char *TextureArray3 = new unsigned char[TextureSize * TextureSize * 4];
    
    void makeTexture()
    {
        for (size_t j=0; j<TextureSize; j++)
        {
            for (size_t i=0; i<TextureSize; i++)
            {
                unsigned char r0 = 0;
                unsigned char g0 = 0;
                unsigned char b0 = 0;
                unsigned char a0 = (unsigned char)255;
                unsigned char r1 = 0;
                unsigned char g1 = 0;
                unsigned char b1 = 0;
                unsigned char a1 = (unsigned char)255;
                unsigned char r2 = 0;
                unsigned char g2 = 0;
                unsigned char b2 = 0;
                unsigned char a2 = (unsigned char)255;
                unsigned char r3 = 0;
                unsigned char g3 = 0;
                unsigned char b3 = 0;
                unsigned char a3 = (unsigned char)255;
                
                int boxSize = 3;
                int ii = i / boxSize;
                int jj = j / boxSize;
                if((ii + jj) % 2 == 0){
                    r0 = (unsigned char)255;
                    g0 = (unsigned char)255;
                    b0 = (unsigned char)255;
                    r1 = (unsigned char)255;
                    g1 = (unsigned char)255;
                    b1 = (unsigned char)255;
                    r2 = (unsigned char)255;
                    g2 = (unsigned char)255;
                    b2 = (unsigned char)255;
                } else {
                    r0 = (unsigned char)( (i + j) / 2 % 256 );
                    g0 = (unsigned char)( (i + j) / 3 % 256 );
                    b0 = (unsigned char)( (i + j) / 4 % 256 );
                    r1 = (unsigned char)( (i + j) / 5 % 256 );
                    g1 = (unsigned char)( (i + j) / 4 % 256 );
                    b1 = (unsigned char)( (i + j) / 3 % 256 );
                    r2 = (unsigned char)( (i + j) / 4 % 256 );
                    g2 = (unsigned char)( (i + j) / 5 % 256 );
                    b2 = (unsigned char)( (i + j) / 6 % 256 );
                }
                if(ii  % 2 == 0){
                    r3 = (unsigned char)255;
                    g3 = (unsigned char)255;
                    b3 = (unsigned char)255;
                } else {
                    r3 = (unsigned char)( (i ) / 4 % 256 );
                    g3 = (unsigned char)( (i ) / 5 % 256 );
                    b3 = (unsigned char)( (i ) / 6 % 256 );
                }
                
                size_t offset = (i + j*TextureSize) * 4;
                TextureArray0[offset + 0] = r0;
                TextureArray0[offset + 1] = g0;
                TextureArray0[offset + 2] = b0;
                TextureArray0[offset + 3] = a0;
                TextureArray1[offset + 0] = r1;
                TextureArray1[offset + 1] = g1;
                TextureArray1[offset + 2] = b1;
                TextureArray1[offset + 3] = a1;
                TextureArray2[offset + 0] = r2;
                TextureArray2[offset + 1] = g2;
                TextureArray2[offset + 2] = b2;
                TextureArray2[offset + 3] = a2;
                TextureArray3[offset + 0] = r3;
                TextureArray3[offset + 1] = g3;
                TextureArray3[offset + 2] = b3;
                TextureArray3[offset + 3] = a3;
            }
        }
    }
    
    // ポリゴンオブジェクトを生成する関数
    kvs::jaea::TexturedPolygonObject* CreateTestData()
    {
        makeTexture();
        
        // KVSの配列クラス（kvs::ValueArray）にセットする。
        kvs::ValueArray<kvs::Real32> coords(CoordArray, N * 3);
        kvs::ValueArray<kvs::UInt8> colors(ColorArray, N * 3);
        kvs::ValueArray<kvs::Real32> normals(NormalArray, N * 3);
        kvs::ValueArray<kvs::UInt32> connections(PolygonConnectionArray, P * 3);
        kvs::ValueArray<kvs::Real32> texture2DCoords(TextureCoordArray, N * 2);
        kvs::ValueArray<kvs::UInt32> textureIds(TextureIdArray, N);
        
        // ポリゴンオブジェクトを生成する。
        kvs::jaea::TexturedPolygonObject* texturedPolygon = new kvs::jaea::TexturedPolygonObject();
        texturedPolygon->setCoords(coords);
        texturedPolygon->setColors(colors);
        texturedPolygon->setNormals(normals);
        texturedPolygon->setConnections(connections);
        texturedPolygon->setPolygonType(kvs::jaea::TexturedPolygonObject::Triangle);
        texturedPolygon->setColorType(kvs::jaea::TexturedPolygonObject::VertexColor);
        texturedPolygon->setNormalType(kvs::jaea::TexturedPolygonObject::VertexNormal);
        texturedPolygon->setTexture2DCoords(texture2DCoords);
        texturedPolygon->setTextureIds(textureIds);
        
        {
            size_t bytesPerChannel = 1;
            int width = TextureSize;
            int height = TextureSize;
            int nChannels = 4;
            kvs::ValueArray<kvs::UInt8> pixels0(TextureArray0, width * height * nChannels);
            kvs::ValueArray<kvs::UInt8> pixels1(TextureArray1, width * height * nChannels);
            kvs::ValueArray<kvs::UInt8> pixels2(TextureArray2, width * height * nChannels);
            kvs::ValueArray<kvs::UInt8> pixels3(TextureArray3, width * height * nChannels);
            
            texturedPolygon->addColorArray(0, pixels0, width, height);
            texturedPolygon->addColorArray(1, pixels1, width, height);
            texturedPolygon->addColorArray(2, pixels2, width, height);
            texturedPolygon->addColorArray(3, pixels3, width, height);
        }
        
        // 生成したポリゴンオブジェクトのポインタを返す。
        return texturedPolygon;
    }
}

int main(int argc, char *argv[])
{
    if(argc <= 1){
        std::cerr << "No arguments." << std::endl;
        return 0;
    }
    
#ifdef USE_OCULUS
    kvs::oculus::jaea::Application app(argc, argv);
    kvs::oculus::jaea::Screen screen(&app);
    
    kvs::oculus::jaea::TouchController event( &screen );
    event.setRotationFactor( 5000.0f );
    event.setTranslationFactor( 5000.0f );
    event.setScalingFactor(30000.0f);
#else
    kvs::glut::Application app( argc, argv );
    kvs::glut::Screen screen( &app );
    screen.setTitle("Example program for kvs::StochasticPolygonRenderer");
#endif
    
    kvs::jaea::TexturedPolygonObject* texturedPolygon = KVSMain::CreateTestData();
    kvs::jaea::StochasticTexturedPolygonRenderer* renderer = new kvs::jaea::StochasticTexturedPolygonRenderer();
    
    std::string option = argv[1];
    if(option == "-nearest"){
        renderer->setNearestInterpolation();
    }else if(option == "-bilinear"){
        renderer->setBilinearInterpolation();
    }else if(option == "-trilinear"){
        renderer->setTrilinearInterpolation();
    }else if(option == "-anisotropic"){
        renderer->setAnisotropicFiltering();
    }else{
        std::cerr << "Unknown option:" << option << std::endl;
        return 0;
    }
    
    screen.registerObject( texturedPolygon, renderer );
    
    screen.show();
    return app.run();
}

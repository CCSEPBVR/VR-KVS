/*****************************************************************************/
/**
 *  @file   FrameEventListener.h
 */
/*----------------------------------------------------------------------------
 *
 *  Copyright (c) Visualization Laboratory, Kyoto University.
 *  All rights reserved.
 *  See http://www.viz.media.kyoto-u.ac.jp/kvs/copyright/ for details.
 *
 *  $Id: PaintEventListener.h 1325 2012-10-04 10:34:52Z naohisa.sakamoto@gmail.com $
 */
/*****************************************************************************/
#ifndef KVS__FRAME_EVENT_LISTENER_H_INCLUDE
#define KVS__FRAME_EVENT_LISTENER_H_INCLUDE

#include <kvs/EventListener>
#include <kvs/EventBase>


namespace kvs
{

/*===========================================================================*/
/**
 *  @brief  FrameEventListener class.
 */
/*===========================================================================*/
class FrameEventListener : public kvs::EventListener
{
public:

    FrameEventListener();
    virtual ~FrameEventListener();

    virtual void update() = 0;

private:

    void onEvent( kvs::EventBase* event = 0 );
};

} // end of namespace kvs

#endif // KVS__FRAME_EVENT_LISTENER_H_INCLUDE
